# walker/main.py
import logging
import queue
import resource
import sys
from pathlib import Path
from typing import Optional, Tuple

import click, attrs
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# Removed imagehash, tqdm, numpy as they are now in Reporter/Indexer

from . import config, database, models
from .models import FileMetadata

def setup_logging():
    """Sets up logging to a file for warnings and errors."""
    log_file = Path(__file__).parent / "walker.log"
    # Configure logging to write to a file, appending to it if it exists.
    # Only messages of level WARNING and above will be logged.
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='a')

sentinel = object()  # A signal to stop the writer thread.

# Default directories to exclude on Windows when scanning a root drive.
# These are case-insensitive.
DEFAULT_WINDOWS_EXCLUDES = [
    "windows",
    "programdata",
    "program files",
    "program files (x86)",
    "pagefile.sys",
    "hiberfil.sys",
    "swapfile.sys",
    "$recycle.bin",
    "system volume information",
    "msocache",
]

# Default directories to exclude on macOS when scanning from the root.
DEFAULT_MACOS_EXCLUDES = [
    "/.DocumentRevisions-V100",
    "/.fseventsd",
    "/.Spotlight-V100",
    "/.Trashes",
    "/private",
    "/dev",
    "/System",
    "/Library",
    "/Applications",
    "/Users/*/Library",
]

def format_bytes(size: int) -> str:
    """Formats a size in bytes to a human-readable string (KB, MB, GB, etc.)."""
    if not isinstance(size, (int, float)):
        return "0 B"
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size >= power and n < len(power_labels) - 1:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"

def db_writer_worker(db_queue: queue.Queue, batch_size: int):
    """
    A dedicated worker that pulls results from the queue and writes them to the DB.
    Uses a batched "upsert" strategy for high performance with SQLite.
    This function creates its own database session to ensure thread safety.
    """
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    def commit_batch(session: Session, current_batch: list):
        """Commits a batch of records to the database."""
        if not current_batch:
            return 0
        
        with database.db_lock:
            try:
                stmt = sqlite_insert(models.FileIndex).values(current_batch)
                update_dict = {c.name: c for c in stmt.excluded if c.name not in ["id", "path"]}
                stmt = stmt.on_conflict_do_update(index_elements=['path'], set_=update_dict)
                session.execute(stmt)
                session.commit()
            except Exception as e:
                # This can happen if the DB is locked by another process.
                if "database is locked" in str(e).lower():
                    click.echo(click.style("\nDatabase is locked by another process. Please close other connections and try again.", fg="red"), err=True)
                    # We can't continue, so we'll re-raise to stop the thread.
                    raise
        count = len(current_batch)
        current_batch.clear()
        return count

    with database.SessionLocal() as db_session:
        click.echo("DB writer worker started.")
        batch = []
        total_processed = 0

        while True:
            item: Optional[FileMetadata] = db_queue.get()
            if item is sentinel:
                total_processed += commit_batch(db_session, batch)
                break

            if item:
                batch.append(attrs.asdict(item))
                if len(batch) >= batch_size:
                    total_processed += commit_batch(db_session, batch)

            db_queue.task_done()

    click.echo(f"DB writer finished. A total of {total_processed} records were written to the database.")

def set_memory_limit(limit_gb: Optional[float]):
    """Sets a soft memory limit for the current process (Linux/macOS only)."""
    if limit_gb is None or sys.platform == "win32":
        return

    try:
        limit_bytes = int(limit_gb * 1024**3)
        # Set both soft and hard limits for virtual memory
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, resource.error) as e:
        # This might fail if the limit is too low or due to permissions.
        # We'll log it but not stop the worker.
        logging.warning(f"Could not set memory limit: {e}")

def process_file_wrapper(path: Path, shared_queue: queue.Queue, memory_limit_gb: Optional[float]) -> Optional[FileMetadata]:
    """Wrapper to instantiate and run the FileProcessor in a separate process/thread."""
    import warnings
    from PIL import Image

    # Filter warnings within the worker process.
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    set_memory_limit(memory_limit_gb)

    # This is where you import the class to avoid pickling issues with some executors
    from .file_processor import FileProcessor
    processor = FileProcessor(path)
    result = processor.process()
    if result:
        shared_queue.put(result)
    return result is not None # Return True if processed, False otherwise

@click.group()
def cli():
    """A powerful file indexer and query tool."""
    pass

@cli.command(name="index")
@click.argument('root_paths', nargs=-1, required=False, type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--workers', default=3, help='Number of processor workers.')
@click.option('--memory-limit', 'memory_limit_gb', type=float, help='Soft memory limit per worker in GB (e.g., 4.0). Linux/macOS only.')
@click.option('--exclude', 'exclude_paths', multiple=True, type=click.Path(), help='Directory name to exclude. Can be used multiple times.')
def index(root_paths: Tuple[Path, ...], workers: int, memory_limit_gb: Optional[float], exclude_paths: Tuple[str, ...]):
    """
    Scans a directory recursively, processes files, and saves metadata to a SQLite DB.

    If ROOT_PATHS are not provided, it will use 'scan_dirs' from walker.toml.
    """
    from .indexer import Indexer

    # Change to the script's directory to reliably find walker.toml and the DB.
    setup_logging()

    script_dir = Path(__file__).parent

    # Load config relative to the script directory
    app_config = config.load_config()

    click.echo(f"Initializing database...")
    database.init_db()

    indexer = Indexer(
        root_paths=root_paths,
        workers=workers,
        memory_limit_gb=memory_limit_gb,
        exclude_paths=exclude_paths,
    )
    indexer.run()

@cli.command(name="download-assets")
def download_assets():
    """
    Downloads and caches all necessary models for offline use.
    """
    from . import download_assets
    download_assets.run_download()

@cli.command(name="find-dupes")
def find_dupes():
    """Finds files with identical content based on their SHA-256 hash."""
    from .reporter import Reporter
    reporter = Reporter()
    reporter.find_dupes()

@cli.command(name="find-image-dupes")
@click.option('--threshold', default=0, type=int, help='Similarity threshold (0-64). 0 finds exact duplicates.')
def find_image_dupes(threshold: int):
    """Finds visually identical or similar images based on perceptual hash."""
    from .reporter import Reporter
    reporter = Reporter()
    reporter.find_image_dupes(threshold)

@cli.command(name="find-similar-text")
@click.option('--threshold', default=0.95, type=click.FloatRange(0.0, 1.0), help='Similarity threshold (0.0 to 1.0).')
def find_similar_text(threshold: float):
    """Finds files with similar text content using vector embeddings."""
    from .reporter import Reporter
    reporter = Reporter()
    reporter.find_similar_text(threshold)

@cli.command(name="search")
@click.argument('query_text', nargs=-1)
@click.option('--limit', default=10, type=int, help='Number of results to return.')
def search(query_text: Tuple[str, ...], limit: int):
    """
    Performs a semantic search for files based on text content.
    
    QUERY_TEXT: The text to search for.
    """
    from .reporter import Reporter
    if not query_text:
        click.echo("Error: Please provide a search query.", err=True)
        return

    full_query = " ".join(query_text)
    reporter = Reporter()
    reporter.search_content(full_query, limit)

@cli.command(name="largest-files")
@click.option('--limit', default=20, type=int, help='Number of files to list.')
def largest_files(limit: int):
    """Lists the largest files in the index by size."""
    from .reporter import Reporter
    reporter = Reporter()
    reporter.largest_files(limit)

@cli.command(name="type-summary")
def type_summary():
    """Shows a summary of file counts and sizes grouped by MIME type."""
    from .reporter import Reporter
    reporter = Reporter()
    reporter.type_summary()

@cli.command(name="list-pii-files")
def list_pii_files():
    """Lists all files that have been flagged for containing potential PII."""
    from .reporter import Reporter
    reporter = Reporter()
    reporter.list_pii_files()

if __name__ == "__main__":
    cli()