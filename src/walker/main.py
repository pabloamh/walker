# walker/main.py
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue, resource
import logging, multiprocessing
import sys
import os, sys
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict

import click, attrs
import imagehash
from sqlalchemy import func
from sqlalchemy.orm import Session
from tqdm import tqdm
import numpy as np

from . import config, database, models, scanner
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
    # Change to the script's directory to reliably find walker.toml and the DB.
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    setup_logging()

    click.echo(f"Working directory set to: {script_dir}")

    app_config = config.load_config()

    click.echo(f"Initializing database...")
    database.init_db()

    # Use a Manager to create a queue that can be shared between processes
    manager = multiprocessing.Manager()
    results_queue = manager.Queue()

    writer_thread = threading.Thread(target=db_writer_worker, args=(results_queue, app_config.db_batch_size))
    writer_thread.start()    # --- Determine which paths to scan ---
    # Combine paths from CLI and config file, then validate and de-duplicate.
    combined_paths = list(root_paths)
    if app_config.scan_dirs:
        click.echo("Adding 'scan_dirs' from walker.toml.")
        for p_str in app_config.scan_dirs:
            p = Path(p_str).expanduser()
            if p not in combined_paths:
                combined_paths.append(p)

    validated_paths = []
    for p in combined_paths:
        p_resolved = p.resolve()
        if p_resolved.is_dir():
            validated_paths.append(p_resolved)
        else:
            click.echo(click.style(f"Warning: Path '{p}' not found or not a directory. Skipping.", fg="yellow"))
    final_root_paths = tuple(sorted(list(set(validated_paths))))
    # --- Merge CLI arguments and config file settings ---
    # CLI options take precedence over the config file.
    final_workers = workers if workers != 3 else app_config.workers
    final_memory_limit = memory_limit_gb if memory_limit_gb is not None else app_config.memory_limit_gb

    # Prepare exclusion list
    final_exclude_list = {path.lower() for path in exclude_paths}

    # --- Apply platform-specific default exclusions for root-level scans ---
    is_root_scan = any(p.parent == p for p in final_root_paths)
    if is_root_scan:
        if sys.platform == "win32":
            click.echo("Windows root drive scan detected. Applying default system exclusions.")
            final_exclude_list.update(DEFAULT_WINDOWS_EXCLUDES)
        elif sys.platform == "darwin": # macOS
            click.echo("macOS root drive scan detected. Applying default system exclusions.")
            # For macOS, we add them as they are, since they include paths.
            # The scanner will handle these absolute paths correctly.
            final_exclude_list.update({p.lower() for p in DEFAULT_MACOS_EXCLUDES})
        elif sys.platform.startswith("linux"):
            click.echo("Linux root drive scan detected. Consider excluding /proc, /sys, /dev, /run.")

    final_exclude_list.update({d.lower() for d in app_config.exclude_dirs})

    if not final_root_paths:
        # If no paths are specified via CLI or config, provide a helpful message.
        config_path = Path("walker.toml")
        if not config_path.is_file():
            click.echo(click.style("Warning: No scan paths provided and 'walker.toml' not found.", fg="yellow"))
            click.echo(f"Please create a '{config_path.resolve()}' file with 'scan_dirs' or specify paths on the command line.")
            click.echo("Example: walker index /path/to/scan")
        else:
            click.echo(click.style("Warning: No scan paths provided.", fg="yellow"))
            click.echo("Please add directories to 'scan_dirs' in your 'walker.toml' or specify paths on the command line.")
        return # Exit the command gracefully

    click.echo(f"Starting scan with {final_workers} workers...")
    if final_memory_limit and sys.platform != "win32":
        click.echo(click.style(f"Applying a soft memory limit of {final_memory_limit:.2f} GB per worker.", fg="blue"))

    def get_file_chunks(chunk_size: int):
        """Yields chunks of file paths from the scanner."""
        chunk = []
        # Create a single generator for all root paths
        all_paths_generator = (
            file_path
            for root in final_root_paths
            for file_path in scanner.scan_directory(root, final_exclude_list)
        )
        for file_path in all_paths_generator:
            chunk.append(file_path)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    # Process files in chunks to keep memory usage low
    with ProcessPoolExecutor(max_workers=final_workers) as executor, \
         database.SessionLocal() as db_session, \
         tqdm(desc="Scanning for files...", unit=" files", postfix={"processed": 0}) as pbar:

        chunk_iterator = get_file_chunks(chunk_size=10000)
        for path_chunk in chunk_iterator:
            pbar.set_description("Filtering files...")
            # --- Smart Update Logic (applied per chunk) ---
            files_to_process_chunk = []
            
            # 1. Pre-cache stats for the current chunk
            scanned_files: Dict[str, Tuple[Path, float]] = {}
            for p in path_chunk:
                try:
                    scanned_files[str(p.resolve())] = (p, p.stat().st_mtime)
                except FileNotFoundError:
                    continue
            
            scanned_paths_set = set(scanned_files.keys())

            # 2. Check this chunk against the database
            existing_files_chunk = {p: m for p, m in db_session.query(models.FileIndex.path, models.FileIndex.mtime).filter(models.FileIndex.path.in_(scanned_paths_set))}
            
            # 3. Find modified files in the chunk
            for path_str, mtime in existing_files_chunk.items():
                if scanned_files[path_str][1] > mtime:
                    files_to_process_chunk.append(scanned_files[path_str][0])
            
            # 4. Find new files in the chunk
            new_paths = scanned_paths_set - set(existing_files_chunk.keys())
            files_to_process_chunk.extend(scanned_files[path_str][0] for path_str in new_paths)

            # Update the total scanned count
            pbar.update(len(path_chunk))

            if not files_to_process_chunk:
                continue

            pbar.set_description("Processing files...")
            # --- Submit the filtered chunk for processing ---
            future_to_path = {
                executor.submit(process_file_wrapper, path, results_queue, final_memory_limit): path for path in files_to_process_chunk
            }

            processed_in_chunk = 0
            for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Processing files", leave=False):
                try:
                    if future.result():
                        processed_in_chunk += 1
                except Exception as exc:
                    path = future_to_path[future]
                    error_message = f"Error processing '{path}': {exc}"
                    logging.error(error_message)
                    tqdm.write(click.style(f"\n{error_message}", fg="red"), file=sys.stderr)

            # Update the main progress bar's postfix after the chunk is done
            current_processed = pbar.postfix["processed"]
            pbar.postfix["processed"] = current_processed + processed_in_chunk
            pbar.refresh()
    # Signal the writer to stop and wait for it to finish
    results_queue.put(sentinel)
    writer_thread.join()

    click.echo("All files have been processed and indexed.")

@cli.command(name="find-dupes")
def find_dupes():
    """Finds files with identical content based on their SHA-256 hash."""
    db_session = database.SessionLocal()
    try:
        click.echo("Querying for duplicate files by hash...")

        # --- Efficiently find and group duplicates in a single query ---
        # Create a subquery to find hashes that have more than one entry.
        dupe_hashes_subq = (
            db_session.query(models.FileIndex.crypto_hash)
            .group_by(models.FileIndex.crypto_hash)
            .having(func.count(models.FileIndex.id) > 1)
            .subquery()
        )

        # Fetch all files that are part of a duplicate set, ordered by hash and then mtime.
        # This ensures that when we group them, the oldest file is first.
        all_dupes = (
            db_session.query(models.FileIndex)
            .filter(models.FileIndex.crypto_hash.in_(dupe_hashes_subq))
            .order_by(models.FileIndex.crypto_hash, models.FileIndex.mtime)
            .all()
        )

        if not all_dupes:
            click.echo("No duplicate files found.")
            return

        # Group the flat list of files by their hash in Python.
        from itertools import groupby
        grouped_dupes = groupby(all_dupes, key=lambda file: file.crypto_hash)

        for i, (hash_val, files_group) in enumerate(grouped_dupes, 1):
            files = list(files_group)
            click.echo(f"\n--- Set {i} ({len(files)} files, hash: {hash_val[:12]}...) ---")
            click.echo(click.style(f"  Source: {files[0].path}", fg="green"))
            for file in files[1:]:
                click.echo(f"  - Dup:  {file.path}")

    finally:
        db_session.close()

@cli.command(name="find-image-dupes")
@click.option('--threshold', default=0, type=int, help='Similarity threshold (0-64). 0 finds exact duplicates.')
def find_image_dupes(threshold: int):
    """Finds visually identical or similar images based on perceptual hash."""
    db_session = database.SessionLocal()
    try:
        click.echo("Querying for duplicate images by perceptual hash...")

        # Fetch all images with a perceptual hash
        images = (
            db_session.query(models.FileIndex.path, models.FileIndex.perceptual_hash)
            .filter(models.FileIndex.perceptual_hash.isnot(None))
            .all()
        )

        if len(images) < 2:
            click.echo("Not enough images in the index to compare.")
            return

        click.echo(f"Comparing {len(images)} images...")

        # Convert phash strings to imagehash objects
        hashes = {path: imagehash.hex_to_hash(phash) for path, phash in images}
        paths = list(hashes.keys())

        # --- Group similar images ---
        groups = []
        if threshold == 0:
            # Fast path for exact duplicates
            from collections import defaultdict
            hash_groups = defaultdict(list)
            for path, phash_str in images:
                hash_groups[phash_str].append(path)
            
            for group_paths in hash_groups.values():
                if len(group_paths) > 1:
                    groups.append(sorted(group_paths))
        else:
            # Efficiently find similar images using NumPy broadcasting
            # 1. Convert hex hashes to a NumPy array of uint8
            hash_array = np.array([h.hash.flatten() for h in hashes.values()], dtype=np.uint8)

            # 2. Calculate Hamming distance matrix efficiently
            # The formula (a != b).sum() is equivalent to the Hamming distance.
            # By using broadcasting, we can compute all pairs at once.
            diff_matrix = np.not_equal(hash_array[np.newaxis, :, :], hash_array[:, np.newaxis, :])
            distance_matrix = np.sum(diff_matrix, axis=2)

            # 3. Find pairs below the threshold
            # We use np.triu with k=1 to get the upper triangle of the matrix, avoiding self-comparisons and duplicates.
            similar_indices = np.argwhere((distance_matrix <= threshold) & (np.triu(np.ones_like(distance_matrix), k=1) == 1))

            # 4. Group the pairs into connected components (sets of similar images)
            from .utils import group_pairs
            path_groups = group_pairs(similar_indices, paths)
            groups = [sorted(g) for g in path_groups]

        if not groups:
            click.echo("No similar images found with the given threshold.")
            return

        for i, group in enumerate(groups, 1):
            click.echo(f"\n--- Similar Group {i} ---")
            for path in group:
                click.echo(f"  - {path}")
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

        # --- Group the pairs into connected components ---
        from .utils import group_pairs
        groups = group_pairs(similar_pairs, paths)

        for i, group in enumerate(groups, 1):
            click.echo(f"\n--- Similar Group {i} ---")
            for path in sorted(group):
                click.echo(f"  - {path}")
    finally:
        db_session.close()

@cli.command(name="search")
@click.argument('query_text', nargs=-1)
@click.option('--limit', default=10, type=int, help='Number of results to return.')
def search(query_text: Tuple[str, ...], limit: int):
    """
    Performs a semantic search for files based on text content.
    
    QUERY_TEXT: The text to search for.
    """
    if not query_text:
        click.echo("Error: Please provide a search query.", err=True)
        return

    full_query = " ".join(query_text)
    db_session = database.SessionLocal()
    try:
        click.echo(f"Searching for files with content similar to: '{full_query}'")

        # This is a simplified implementation. For larger databases, consider
        # a dedicated vector search index like FAISS.
        from .file_processor import embedding_model
        from sklearn.metrics.pairwise import cosine_similarity

        # 1. Generate embedding for the user's query
        query_embedding = embedding_model.encode([full_query])

        # 2. Fetch all file embeddings from the database
        results = db_session.query(models.FileIndex.path, models.FileIndex.content_embedding).filter(
            models.FileIndex.content_embedding.isnot(None)
        ).all()

        if not results:
            click.echo("No text files with embeddings found in the index.")
            return

        paths = [r.path for r in results]
        file_embeddings = np.array([np.frombuffer(r.content_embedding, dtype=np.float32) for r in results])

        # 3. Calculate similarity and rank results
        similarities = cosine_similarity(query_embedding, file_embeddings)[0]
        top_indices = np.argsort(similarities)[-limit:][::-1]

        click.echo(f"\n--- Top {limit} results ---")
        for i in top_indices:
            click.echo(f"Score: {similarities[i]:.4f} | {paths[i]}")

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

@cli.command(name="list-pii-files")
def list_pii_files():
    """Lists all files that have been flagged for containing potential PII."""
    db_session = database.SessionLocal()
    try:
        click.echo("Querying for files flagged with potential PII...")
        
        files = (
            db_session.query(models.FileIndex)
            .filter(models.FileIndex.has_pii == True)
            .order_by(models.FileIndex.path)
            .all()
        )

        if not files:
            click.echo("No files containing potential PII were found.")
            return

        for file in files:
            click.echo(file.path)
    finally:
        db_session.close()

if __name__ == "__main__":
    cli()