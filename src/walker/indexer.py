# walker/indexer.py
import logging
import multiprocessing
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import click
from sqlalchemy.orm import Session
from tqdm import tqdm

from . import config, database, models, scanner
from . import worker


class Indexer:
    """Orchestrates the file indexing process."""

    def __init__(
        self,
        root_paths: Tuple[Path, ...],
        workers: int,
        memory_limit_gb: Optional[float],
        exclude_paths: Tuple[str, ...],
    ):
        self.app_config = config.load_config()
        self.cli_root_paths = root_paths
        self.cli_workers = workers
        self.cli_memory_limit_gb = memory_limit_gb
        self.cli_exclude_paths = exclude_paths

        self.final_root_paths: Tuple[Path, ...] = tuple()
        self.final_workers: int = 0
        self.final_memory_limit: Optional[float] = None
        self.final_exclude_list: set[str] = set()

    def _prepare_paths_and_exclusions(self):
        """Validates paths and merges exclusions from CLI and config."""
        # --- Determine which paths to scan ---
        combined_paths = list(self.cli_root_paths)
        if self.app_config.scan_dirs:
            click.echo("Adding 'scan_dirs' from walker.toml.")
            for p_str in self.app_config.scan_dirs:
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
        self.final_root_paths = tuple(sorted(list(set(validated_paths))))

        # --- Prepare exclusion list ---
        self.final_exclude_list = {path.lower() for path in self.cli_exclude_paths}
        self.final_exclude_list.update({d.lower() for d in self.app_config.exclude_dirs})

        # --- Apply platform-specific default exclusions for root-level scans ---
        is_root_scan = any(p.parent == p for p in self.final_root_paths)
        if is_root_scan:
            if sys.platform == "win32":
                click.echo("Windows root drive scan detected. Applying default system exclusions.")
                self.final_exclude_list.update(models.DEFAULT_WINDOWS_EXCLUDES)
            elif sys.platform == "darwin":  # macOS
                click.echo("macOS root drive scan detected. Applying default system exclusions.")
                self.final_exclude_list.update({p.lower() for p in models.DEFAULT_MACOS_EXCLUDES})
            elif sys.platform.startswith("linux"):
                click.echo("Linux root drive scan detected. Consider excluding /proc, /sys, /dev, /run.")

    def _prepare_settings(self):
        """Merges CLI arguments and config file settings."""
        self.final_workers = self.cli_workers if self.cli_workers != 3 else self.app_config.workers
        self.final_memory_limit = self.cli_memory_limit_gb if self.cli_memory_limit_gb is not None else self.app_config.memory_limit_gb

    def _get_file_chunks(self, chunk_size: int):
        """Yields chunks of file paths from the scanner."""
        chunk = []
        all_paths_generator = (
            file_path
            for root in self.final_root_paths
            for file_path in scanner.scan_directory(root, self.final_exclude_list)
        )
        for file_path in all_paths_generator:
            chunk.append(file_path)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _filter_chunk(self, path_chunk: list[Path], existing_files: Dict[str, float]) -> list[Path]:
        """Filters a chunk of paths against an in-memory dictionary of existing files."""
        files_to_process = []
        for p in path_chunk:
            try:
                path_str = str(p.resolve())
                mtime = p.stat().st_mtime
                if path_str not in existing_files or mtime > existing_files[path_str]:
                    files_to_process.append(p)
            except FileNotFoundError:
                continue
        return files_to_process

    def run(self):
        """Executes the entire indexing process."""
        self._prepare_settings()
        self._prepare_paths_and_exclusions()

        if not self.final_root_paths:
            config_path = Path("walker.toml")
            if not config_path.is_file():
                click.echo(click.style("Warning: No scan paths provided and 'walker.toml' not found.", fg="yellow"))
                click.echo(f"Please create a '{config_path.resolve()}' file with 'scan_dirs' or specify paths on the command line.")
            else:
                click.echo(click.style("Warning: No scan paths provided.", fg="yellow"))
                click.echo("Please add directories to 'scan_dirs' in your 'walker.toml' or specify paths on the command line.")
            return

        click.echo(f"Starting scan with {self.final_workers} workers...")
        if self.final_memory_limit and sys.platform != "win32":
            click.echo(click.style(f"Applying a soft memory limit of {self.final_memory_limit:.2f} GB per worker.", fg="blue"))

        manager = multiprocessing.Manager()
        results_queue = manager.Queue()
        writer_thread = threading.Thread(target=worker.db_writer_worker, args=(results_queue, self.app_config.db_batch_size))
        writer_thread.start()

        with ProcessPoolExecutor(max_workers=self.final_workers) as executor, \
             tqdm(desc="Scanning for files...", unit=" files", postfix={"processed": 0}) as pbar:

            # --- Pre-load existing file index into memory for fast lookups ---
            click.echo("Loading existing file index into memory...")
            with database.get_session() as db_session:
                existing_files_from_db = db_session.query(models.FileIndex.path, models.FileIndex.mtime).all()
                existing_files_map = {path: mtime for path, mtime in existing_files_from_db}
            click.echo(f"Loaded {len(existing_files_map)} records from the index.")

            chunk_iterator = self._get_file_chunks(chunk_size=10000)
            for path_chunk in chunk_iterator:
                pbar.set_description("Filtering files...")
                files_to_process = self._filter_chunk(path_chunk, existing_files_map)
                pbar.update(len(path_chunk))

                if not files_to_process:
                    continue

                pbar.set_description("Processing files...")
                future_to_path = {
                    executor.submit(worker.process_file_wrapper, path, results_queue, self.final_memory_limit): path
                    for path in files_to_process
                }

                processed_in_chunk = 0
                for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Processing chunk", leave=False):
                    try:
                        if future.result():
                            processed_in_chunk += 1
                    except Exception as exc:
                        path = future_to_path[future]
                        error_message = f"Error processing '{path}': {exc}"
                        logging.error(error_message)
                        tqdm.write(click.style(f"\n{error_message}", fg="red"), file=sys.stderr)

                current_processed = pbar.postfix["processed"]
                pbar.postfix["processed"] = current_processed + processed_in_chunk
                pbar.refresh()

        results_queue.put(worker.sentinel)
        writer_thread.join()
        click.echo("All files have been processed and indexed.")