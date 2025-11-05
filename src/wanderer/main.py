# walker/main.py
import logging
from pathlib import Path
from typing import Optional, Tuple

import click

from . import config, database, models, log_manager

def setup_logging():
    """Sets up logging to a file for warnings and errors."""
    log_file = Path(__file__).parent / "wanderer.log"
    # Use a custom handler to prevent huge log files from repetitive errors.
    handler = log_manager.DeduplicatingLogHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Configure the root logger
    logging.basicConfig(level=logging.WARNING, handlers=[handler])

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

    If ROOT_PATHS are not provided, it will use 'scan_dirs' from wanderer.toml.
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

@cli.command(name="refine-unknowns")
@click.option('--workers', default=3, help='Number of processor workers.')
def refine_unknowns(workers: int):
    """
    Post-processes files with generic MIME types using Fido.

    This command finds files indexed as 'application/octet-stream' or
    'inode/x-empty' and re-processes them with Fido to get a more
    accurate PRONOM ID and MIME type. Requires 'use_fido = true' in config.
    """
    from .indexer import Indexer
    indexer = Indexer(root_paths=(), workers=workers, memory_limit_gb=None, exclude_paths=())
    indexer.refine_unknown_files()

@cli.command(name="refine-text")
@click.option('--workers', default=3, help='Number of processor workers.')
def refine_text(workers: int):
    """
    Post-processes text-based files to extract content and detect PII.

    This command finds files that are likely to contain text (based on MIME
    type) but do not yet have their content stored in the database. It is
    useful if you ran the initial index with 'extract_text_on_scan = false'
    in your configuration to speed up scanning.
    """
    from .indexer import Indexer
    # Force text extraction to be on for this operation.
    indexer = Indexer(root_paths=(), workers=workers, memory_limit_gb=None, exclude_paths=())
    indexer.refine_text_content()

@cli.command(name="refine-text-by-path")
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--workers', default=3, help='Number of processor workers.')
def refine_text_by_path(paths: Tuple[Path, ...], workers: int):
    """
    Extracts text content for all files under a specific path.

    This is useful for performing a deep text analysis on a targeted
    directory (e.g., '~/Documents') after a fast initial scan.
    """
    from .indexer import Indexer
    indexer = Indexer(root_paths=(), workers=workers, memory_limit_gb=None, exclude_paths=())
    indexer.refine_text_content_by_path(paths)

@cli.command(name="refine-images-by-path")
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--workers', default=3, help='Number of processor workers.')
def refine_images_by_path(paths: Tuple[Path, ...], workers: int):
    """
    Computes perceptual hashes for all images under a specific path.

    This is useful for enabling similar-image search on a targeted
    directory (e.g., '~/Pictures') after a fast initial scan.
    """
    from .indexer import Indexer
    indexer = Indexer(root_paths=(), workers=workers, memory_limit_gb=None, exclude_paths=())
    indexer.refine_image_content_by_path(paths)

@cli.command(name="refine-fido-by-path")
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--workers', default=3, help='Number of processor workers.')
def refine_fido_by_path(paths: Tuple[Path, ...], workers: int):
    """
    Forces a Fido rescan for all files under a specific path.

    This is useful for ensuring the highest accuracy file identification
    on a targeted directory (e.g., '~/Scans').
    """
    from .indexer import Indexer
    indexer = Indexer(root_paths=(), workers=workers, memory_limit_gb=None, exclude_paths=())
    indexer.refine_fido_by_path(paths)

@cli.command(name="refine-images-by-mime-and-path")
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--mime-type', 'mime_types', multiple=True, required=True, help='MIME type to target (e.g., "image/jpeg"). Can be used multiple times.')
@click.option('--workers', default=3, help='Number of processor workers.')
def refine_images_by_mime_and_path(paths: Tuple[Path, ...], mime_types: Tuple[str, ...], workers: int):
    """
    Computes perceptual hashes for specific image MIME types under a path.

    Example: phash for jpegs under '~/Pictures':
    ... refine-images-by-mime-and-path --mime-type "image/jpeg" ~/Pictures
    """
    from .indexer import Indexer
    indexer = Indexer(root_paths=(), workers=workers, memory_limit_gb=None, exclude_paths=())
    indexer.refine_image_content_by_path(paths, mime_types=mime_types)

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

@cli.command(name="pronom-summary")
def pronom_summary():
    """Shows a summary of file counts grouped by PRONOM ID."""
    from .reporter import Reporter
    reporter = Reporter()
    reporter.pronom_summary()

@cli.command(name="lonely-files")
@click.option('--limit', default=20, type=int, help='Number of files to list.')
def lonely_files(limit: int):
    """
    Lists unique files that have no content duplicates.
    This is useful for finding unique assets in your collection.
    """
    from .reporter import Reporter
    reporter = Reporter()
    reporter.find_lonely_files(limit)

@cli.command(name="list-pii-files")
def list_pii_files():
    """Lists all files that have been flagged for containing potential PII."""
    from .reporter import Reporter
    reporter = Reporter()
    reporter.list_pii_files()

@cli.command(name="gui")
def gui():
    """Launches the Wanderer graphical user interface."""
    from . import gui
    import flet as ft
    database.init_db()
    ft.app(target=gui.main)

@cli.command(name="gui-qt")
def gui_qt():
    """Launches a sample Wanderer GUI using PyQt/PySide."""
    from . import gui_qt
    database.init_db()
    gui_qt.main_qt() # type: ignore

if __name__ == "__main__":
    cli()