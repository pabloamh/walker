# walker/indexer.py
import concurrent.futures
import logging
import queue
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple, Set, List

from . import config, database, models, scanner
from .models import FileMetadata

# A queue to hold file processing results before they are written to the DB.
results_queue: queue.Queue[Optional[FileMetadata]] = queue.Queue()
sentinel = object()  # A signal to stop the writer thread.


def db_writer_worker(batch_size: int, progress_callback: Callable[[str], None]):
    """
    A dedicated worker that pulls results from the queue and writes them to the DB.
    """
    db_session = database.SessionLocal()
    progress_callback("DB writer worker started.")
    batch = []
    total_processed = 0
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    while True:
        item = results_queue.get()
        if item is sentinel:
            if batch:
                with database.db_lock:
                    stmt = sqlite_insert(models.FileIndex).values(batch)
                    update_dict = {c.name: c for c in stmt.excluded if c.name not in ["id", "path"]}
                    stmt = stmt.on_conflict_do_update(index_elements=['path'], set_=update_dict)
                    db_session.execute(stmt)
                    db_session.commit()
                    total_processed += len(batch)
            break

        if item:
            import attrs
            batch.append(attrs.asdict(item))
            if len(batch) >= batch_size:
                with database.db_lock:
                    stmt = sqlite_insert(models.FileIndex).values(batch)
                    update_dict = {c.name: c for c in stmt.excluded if c.name not in ["id", "path"]}
                    stmt = stmt.on_conflict_do_update(index_elements=['path'], set_=update_dict)
                    db_session.execute(stmt)
                    db_session.commit()
                    total_processed += len(batch)
                    progress_callback(f"DB writer committed {len(batch)} records.")
                    batch.clear()

        results_queue.task_done()

    db_session.close()
    progress_callback(f"DB writer finished. Processed {total_processed} files.")


def process_file_wrapper(path: Path) -> Optional[FileMetadata]:
    """Wrapper to instantiate and run the FileProcessor in a separate process/thread."""
    from .file_processor import FileProcessor
    processor = FileProcessor(path)
    return processor.process()


class Indexer:
    def __init__(
        self,
        root_paths: Tuple[Path, ...],
        workers: int,
        exclude_paths: Tuple[str, ...],
        progress_callback: Callable[[str], None],
        progress_bar_callback: Optional[Callable] = None,
    ):
        self.root_paths = root_paths
        self.workers = workers
        self.exclude_paths = exclude_paths
        self.progress_callback = progress_callback
        self.progress_bar_callback = progress_bar_callback

    def run(self):
        """Runs the complete indexing process."""
        app_config = config.load_config()

        self.progress_callback("Initializing database...")
        database.init_db()
        writer_thread = threading.Thread(target=db_writer_worker, args=(app_config.db_batch_size, self.progress_callback))
        writer_thread.start()

        # --- Determine which paths to scan ---
        combined_paths = list(self.root_paths)
        if not self.root_paths and app_config.scan_dirs:
            self.progress_callback("Using 'scan_dirs' from walker.toml.")
            for p_str in app_config.scan_dirs:
                p = Path(p_str).expanduser()
                if p not in combined_paths:
                    combined_paths.append(p)

        validated_paths = []
        for p in combined_paths:
            if p.is_dir():
                validated_paths.append(p.resolve())
            else:
                self.progress_callback(f"Warning: Path '{p}' not found or not a directory. Skipping.")
        final_root_paths = tuple(sorted(list(set(validated_paths))))

        # --- Prepare exclusion list ---
        final_exclude_list = {path.lower() for path in self.exclude_paths}
        final_exclude_list.update({d.lower() for d in app_config.exclude_dirs})

        if not final_root_paths:
            self.progress_callback("Error: No valid scan paths provided.")
            return

        self.progress_callback(f"Starting scan with {self.workers} workers...")

        all_file_paths = []
        for path in final_root_paths:
            for file_path in scanner.scan_directory(path, final_exclude_list):
                all_file_paths.append(file_path)
        self.progress_callback(f"Found {len(all_file_paths)} total files.")

        # --- Smart Update Logic ---
        self.progress_callback("Checking for new or modified files...")
        db_session = database.SessionLocal()
        existing_files = {p: m for p, m in db_session.query(models.FileIndex.path, models.FileIndex.mtime)}

        files_to_process = []
        for path in all_file_paths:
            path_str = str(path.resolve())
            current_mtime = path.stat().st_mtime
            if path_str not in existing_files or current_mtime > existing_files[path_str]:
                files_to_process.append(path)
        db_session.close()

        self.progress_callback(f"Found {len(files_to_process)} new or modified files to process.")
        if not files_to_process:
            results_queue.put(sentinel)
            writer_thread.join()
            self.progress_callback("Indexing complete. No new files to process.")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_path = {executor.submit(process_file_wrapper, path): path for path in files_to_process}

            iterator = concurrent.futures.as_completed(future_to_path)
            if self.progress_bar_callback:
                iterator = self.progress_bar_callback(iterator, total=len(files_to_process), desc="Processing files")

            for future in iterator:
                try:
                    result = future.result()
                    if result:
                        results_queue.put(result)
                except Exception as exc:
                    path = future_to_path[future]
                    error_message = f"Error processing '{path}': {exc}"
                    logging.error(error_message)
                    self.progress_callback(f"ERROR: {error_message}")

        results_queue.join()
        results_queue.put(sentinel)
        writer_thread.join()
        self.progress_callback("All files have been processed and indexed.")