# walker/main.py
import concurrent.futures
import queue
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import click
from sqlalchemy import func
from sqlalchemy.orm import Session
from tqdm import tqdm
import numpy as np

from . import config, database, models, scanner
from .models import FileMetadata

# A queue to hold file processing results before they are written to the DB.
results_queue = queue.Queue()
sentinel = object()  # A signal to stop the writer thread.

# Default directories to exclude on Windows when scanning a root drive.
# These are case-insensitive.
DEFAULT_WINDOWS_EXCLUDES = [
    "windows",
    "program files",
    "program files (x86)",
    "$recycle.bin",
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

def db_writer_worker(db_session: Session, batch_size: int = 100):
    """
    A dedicated worker that pulls results from the queue and writes them to the DB.
    Writes are batched to improve performance.
    """
    print("DB writer worker started.")
    new_files_count = 0
    updated_files_count = 0
    batch = []
    while True:
        item: Optional[FileMetadata] = results_queue.get()
        if item is sentinel: # End of queue
            if batch: # Commit any remaining items in the batch
                db_session.bulk_update_mappings(models.FileIndex, batch)
                db_session.commit()
                updated_files_count += len(batch)
            break

        if item:
            # This is an update, so we add it to the batch
            batch.append(attrs.asdict(item))
            if len(batch) >= batch_size:
                db_session.bulk_update_mappings(models.FileIndex, batch)
                db_session.commit()
                updated_files_count += len(batch)
                batch.clear()
        elif item is not None: # A new file to be added
            # For simplicity, we'll handle new files individually for now
            # A more advanced implementation could batch these as well
            new_files_count += 1
        results_queue.task_done()
    
    print(f"DB writer finished. Updated {updated_files_count} and added {new_files_count} files.")

def process_file_wrapper(path: Path) -> Optional[FileMetadata]:
    """Wrapper to instantiate and run the FileProcessor in a separate process/thread."""
    # This is where you import the class to avoid pickling issues with some executors
    from .file_processor import FileProcessor
    processor = FileProcessor(path)
    return processor.process()

@click.group()
def cli():
    """A powerful file indexer and query tool."""
    pass

@cli.command(name="index")
@click.argument('root_paths', nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--workers', default=3, help='Number of processor workers.')
@click.option('--exclude', 'exclude_paths', multiple=True, type=click.Path(), help='Directory name to exclude. Can be used multiple times.')
def index(root_paths: Tuple[Path, ...], workers: int, exclude_paths: Tuple[str, ...]):
    """
    Scans a directory recursively, processes files, and saves metadata to a SQLite DB.

    ROOT_PATHS: One or more directories to start scanning from.
    """
    click.echo(f"Initializing database...")
    app_config = config.load_config()

    database.init_db()
    db_session = database.SessionLocal()

    # Start the dedicated database writer thread
    writer_thread = threading.Thread(target=db_writer_worker, args=(db_session,))
    writer_thread.start()

    # --- Merge CLI arguments and config file settings ---
    # CLI options take precedence over the config file.
    final_workers = workers if workers != 3 else app_config.workers

    # Prepare exclusion list
    final_exclude_list = {path.lower() for path in exclude_paths}
    is_windows_root_scan = sys.platform == "win32" and any(
        p.parent == p for p in root_paths
    )

    if is_windows_root_scan:
        click.echo("Windows root drive scan detected. Applying default system exclusions.")
        final_exclude_list.update(DEFAULT_WINDOWS_EXCLUDES)
    final_exclude_list.update({d.lower() for d in app_config.exclude_dirs})

    click.echo(f"Starting scan with {final_workers} workers...")
    
    all_file_paths = []
    with tqdm(desc="Scanning directories", unit=" files") as pbar:
        for path in root_paths:
            for file_path in scanner.scan_directory(path, final_exclude_list):
                all_file_paths.append(file_path)
                pbar.update(1)
    
    click.echo(f"Found {len(all_file_paths)} total files.")
    
    # --- Smart Update Logic ---
    click.echo("Checking for new or modified files...")
    existing_files = {
        p: m for p, m in db_session.query(models.FileIndex.path, models.FileIndex.mtime)
    }
    
    file_paths_to_process = []
    for path in tqdm(all_file_paths, desc="Filtering files"):
        path_str = str(path)
        current_mtime = path.stat().st_mtime
        if path_str not in existing_files or current_mtime > existing_files[path_str]:
            file_paths_to_process.append(path)

    file_paths = list(set(file_paths_to_process)) # Remove duplicates
    click.echo(f"Found {len(file_paths)} new or modified files to process.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=final_workers) as executor:
        future_to_path = {executor.submit(process_file_wrapper, path): path for path in file_paths}

        progress_bar = tqdm(concurrent.futures.as_completed(future_to_path), total=len(file_paths), desc="Processing files")
        for future in progress_bar:
            try:
                result = future.result()
                if result:
                    results_queue.put(result)
            except Exception as exc:
                path = future_to_path[future]
                click.echo(f"\n'{path}' generated an exception: {exc}", err=True)

    # Signal the writer to stop and wait for it to finish
    results_queue.put(sentinel)
    writer_thread.join()
    db_session.close()
    click.echo("All files have been processed and indexed.")

@cli.command(name="find-dupes")
def find_dupes():
    """Finds files with identical content based on their SHA-256 hash."""
    db_session = database.SessionLocal()
    try:
        click.echo("Querying for duplicate files by hash...")
        
        # Find hashes that appear more than once
        duplicate_hashes = (
            db_session.query(models.FileIndex.crypto_hash, func.count(models.FileIndex.id).label("count"))
            .group_by(models.FileIndex.crypto_hash)
            .having(func.count(models.FileIndex.id) > 1)
            .all()
        )

        if not duplicate_hashes:
            click.echo("No duplicate files found.")
            return

        click.echo(f"Found {len(duplicate_hashes)} sets of duplicate files.")
        
        for i, (hash_val, count) in enumerate(duplicate_hashes, 1):
            click.echo(f"\n--- Set {i} ({count} files, hash: {hash_val[:12]}...) ---")
            files = db_session.query(models.FileIndex).filter_by(crypto_hash=hash_val).all()
            for file in files:
                click.echo(file.path)

    finally:
        db_session.close()

@cli.command(name="find-image-dupes")
def find_image_dupes():
    """Finds visually identical images based on their perceptual hash."""
    db_session = database.SessionLocal()
    try:
        click.echo("Querying for duplicate images by perceptual hash...")
        
        duplicate_phashes = (
            db_session.query(models.FileIndex.perceptual_hash, func.count(models.FileIndex.id).label("count"))
            .filter(models.FileIndex.perceptual_hash.isnot(None))
            .group_by(models.FileIndex.perceptual_hash)
            .having(func.count(models.FileIndex.id) > 1)
            .all()
        )

        if not duplicate_phashes:
            click.echo("No duplicate images found.")
            return

        click.echo(f"Found {len(duplicate_phashes)} sets of duplicate images.")
        
        for i, (phash_val, count) in enumerate(duplicate_phashes, 1):
            click.echo(f"\n--- Set {i} ({count} images, p-hash: {phash_val}) ---")
            files = db_session.query(models.FileIndex).filter_by(perceptual_hash=phash_val).all()
            for file in files:
                click.echo(file.path)
    finally:
        db_session.close()

@cli.command(name="find-similar-text")
@click.option('--threshold', default=0.95, type=click.FloatRange(0.0, 1.0), help='Similarity threshold (0.0 to 1.0).')
def find_similar_text(threshold: float):
    """Finds files with similar text content using vector embeddings."""
    db_session = database.SessionLocal()
    try:
        click.echo("Querying for files with text content...")
        
        # Fetch all entries that have a content embedding
        results = db_session.query(models.FileIndex.path, models.FileIndex.content_embedding).filter(
            models.FileIndex.content_embedding.isnot(None)
        ).all()

        if len(results) < 2:
            click.echo("Not enough text files in the index to compare.")
            return

        click.echo(f"Found {len(results)} text files. Calculating similarities...")

        paths = [r.path for r in results]
        # The embedding dimension for 'all-MiniLM-L6-v2' is 384
        embeddings = np.array([np.frombuffer(r.content_embedding, dtype=np.float32) for r in results])

        # Use scikit-learn for efficient cosine similarity calculation
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Find pairs above the threshold
        similar_pairs = np.argwhere(np.triu(similarity_matrix, k=1) >= threshold)

        if similar_pairs.shape[0] == 0:
            click.echo("No similar text files found above the threshold.")
            return

        # Group similar files
        # This is a simple grouping method; more advanced clustering could be used
        processed_indices = set()
        group_id = 1
        for i, j in similar_pairs:
            if i not in processed_indices or j not in processed_indices:
                # Find all items similar to the first item in the pair
                similar_to_i = {idx for idx, sim in enumerate(similarity_matrix[i]) if sim >= threshold}
                if not similar_to_i.issubset(processed_indices):
                    click.echo(f"\n--- Similar Group {group_id} ---")
                    for idx in sorted(list(similar_to_i)):
                        click.echo(paths[idx])
                        processed_indices.add(idx)
                    group_id += 1
    finally:
        db_session.close()

@cli.command(name="largest-files")
@click.option('--limit', default=20, type=int, help='Number of files to list.')
def largest_files(limit: int):
    """Lists the largest files in the index by size."""
    db_session = database.SessionLocal()
    try:
        click.echo(f"Querying for the {limit} largest files...")
        
        files = (
            db_session.query(models.FileIndex)
            .order_by(models.FileIndex.size_bytes.desc())
            .limit(limit)
            .all()
        )

        if not files:
            click.echo("No files found in the index.")
            return

        for file in files:
            click.echo(f"{format_bytes(file.size_bytes):>10} | {file.path}")

    finally:
        db_session.close()

@cli.command(name="type-summary")
def type_summary():
    """Shows a summary of file counts and sizes grouped by MIME type."""
    db_session = database.SessionLocal()
    try:
        click.echo("Generating file type summary...")
        
        summary = (
            db_session.query(
                models.FileIndex.mime_type,
                func.count(models.FileIndex.id).label("count"),
                func.sum(models.FileIndex.size_bytes).label("total_size")
            )
            .group_by(models.FileIndex.mime_type)
            .order_by(func.count(models.FileIndex.id).desc())
            .all()
        )

        if not summary:
            click.echo("No files found in the index.")
            return

        click.echo(f"{'MIME Type':<60} | {'Count':>10} | {'Total Size':>12}")
        click.echo("-" * 86)
        for mime_type, count, total_size in summary:
            click.echo(f"{str(mime_type):<60} | {count:>10} | {format_bytes(total_size):>12}")
    finally:
        db_session.close()

if __name__ == "__main__":
    cli()