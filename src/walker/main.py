# walker/main.py
import concurrent.futures
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple

import click
from sqlalchemy.orm import Session
from tqdm import tqdm

from . import database, models, scanner
from .models import FileMetadata

# A queue to hold file processing results before they are written to the DB.
results_queue = queue.Queue()
sentinel = object()  # A signal to stop the writer thread.

def db_writer_worker(db_session: Session):
    """
    A dedicated worker that pulls results from the queue and writes them to the DB.
    """
    print("DB writer worker started.")
    processed_count = 0
    while True:
        item: FileMetadata = results_queue.get()
        if item is sentinel:
            break

        if item:
            with database.db_lock:
                exists = db_session.query(models.FileIndex).filter_by(path=item.path).first()
                if not exists:
                    db_entry = models.FileIndex.from_metadata(item)
                    db_session.add(db_entry)
                    db_session.commit()
                    processed_count += 1
        results_queue.task_done()
    
    print(f"DB writer finished. Added {processed_count} new files to the index.")

def process_file_wrapper(path: Path) -> Optional[FileMetadata]:
    """Wrapper to instantiate and run the FileProcessor in a separate process/thread."""
    # This is where you import the class to avoid pickling issues with some executors
    from .file_processor import FileProcessor
    processor = FileProcessor(path)
    return processor.process()

@click.command()
@click.argument('root_paths', nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--workers', default=3, help='Number of processor workers.')
def run_indexer(root_paths: Tuple[Path, ...], workers: int):
    """
    Scans a directory recursively, processes files, and saves metadata to a SQLite DB.

    ROOT_PATHS: One or more directories to start scanning from.
    """
    click.echo(f"Initializing database...")
    database.init_db()
    db_session = database.SessionLocal()

    # Start the dedicated database writer thread
    writer_thread = threading.Thread(target=db_writer_worker, args=(db_session,))
    writer_thread.start()

    click.echo(f"Scanning directories: {', '.join(map(str, root_paths))}...")
    
    all_file_paths = []
    for path in root_paths:
        all_file_paths.extend(scanner.scan_directory(path))
    file_paths = list(set(all_file_paths)) # Use set to remove duplicates if paths overlap
    click.echo(f"Found {len(file_paths)} files to process.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
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

if __name__ == "__main__":
    run_indexer()