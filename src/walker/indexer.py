# walker/indexer.py
import logging
import multiprocessing
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple
import os
import click
from sqlalchemy.orm import Session
from tqdm import tqdm

from . import config, database, models, scanner
from . import worker


class Indexer:
    """Orchestrates the file scanning, processing, and indexing workflow."""

    def __init__(
        self,
        root_paths: Tuple[Path, ...],
        workers: int,
        memory_limit_gb: Optional[float],
        exclude_paths: Tuple[str, ...],
    ):
        """
        Initializes the Indexer with CLI arguments and application configuration.

        Args:
            root_paths: A tuple of root directories to start scanning from.
            workers: The number of worker processes to use.
            memory_limit_gb: An optional soft memory limit for each worker in GB.
            exclude_paths: A tuple of paths or patterns to exclude from the scan.
        """
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
        """
        Validates root paths and merges exclusion lists from the CLI and config file.
        Also applies platform-specific default exclusions for root-level scans.
        """
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
        # Normalize all exclusion paths to be case-insensitive and use the OS's path separator.
        # This ensures consistent matching across platforms.
        self.final_exclude_list = {os.path.normcase(path) for path in self.cli_exclude_paths}
        self.final_exclude_list.update({os.path.normcase(d) for d in self.app_config.exclude_dirs})

        # --- Apply platform-specific default exclusions for all scans ---
        # This is a safety measure to prevent accidental scanning of sensitive system files.
        if sys.platform == "win32":
            click.echo("Applying default Windows system exclusions.")
            self.final_exclude_list.update({os.path.normcase(p) for p in models.DEFAULT_WINDOWS_EXCLUDES})
        elif sys.platform == "darwin":  # macOS
            click.echo("Applying default macOS system exclusions.")
            self.final_exclude_list.update({os.path.normcase(p) for p in models.DEFAULT_MACOS_EXCLUDES})
        elif sys.platform.startswith("linux"):
            if any(p.parent == p for p in self.final_root_paths):
                click.echo(click.style("Linux root scan detected. Consider excluding /proc, /sys, /dev, /run for safety.", fg="yellow"))

    def _prepare_settings(self):
        """
        Merges settings from CLI arguments and the configuration file.
        CLI arguments take precedence over config file settings.
        """
        self.final_workers = self.cli_workers if self.cli_workers != 3 else self.app_config.workers
        self.final_memory_limit = self.cli_memory_limit_gb if self.cli_memory_limit_gb is not None else self.app_config.memory_limit_gb

    def _get_file_chunks(self, chunk_size: int):
        """
        Scans the configured root paths and yields chunks of file paths.

        Args:
            chunk_size: The maximum number of paths to include in each chunk.

        Yields:
            A list of Path objects.
        """
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
        """
        Filters a chunk of paths to find new or modified files.

        This compares files against an in-memory dictionary of existing file paths
        and their modification times.

        Args:
            path_chunk: A list of file paths to filter.
            existing_files: A dictionary mapping file paths to their mtime.

        Returns:
            A list of paths for files that need to be processed.
        """
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

    def _execute_processing_pool(self, paths_to_process: list[Path], description: str) -> None:
        """
        Manages a process pool to process a list of files and write results to the DB.

        Args:
            paths_to_process: A list of file paths to be processed by the workers.
            description: A description for the tqdm progress bar.
        """
        manager = multiprocessing.Manager()
        results_queue = manager.Queue()
        writer_thread = threading.Thread(target=worker.db_writer_worker, args=(results_queue, self.app_config.db_batch_size))
        writer_thread.start()

        with ProcessPoolExecutor(max_workers=self.final_workers) as executor:
            future_to_path = {
                executor.submit(worker.process_file_wrapper, path, self.app_config, results_queue, self.final_memory_limit): path
                for path in paths_to_process
            }

            WORKER_TIMEOUT_SECONDS = 300  # 5 minutes
            for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc=description):
                try:
                    future.result(timeout=WORKER_TIMEOUT_SECONDS)
                except TimeoutError:
                    path = future_to_path[future]
                    error_message = f"Worker timed out after {WORKER_TIMEOUT_SECONDS}s processing '{path}'. It may be a very large or corrupt file."
                    logging.error(error_message)
                    tqdm.write(click.style(f"\n{error_message}", fg="red"), file=sys.stderr)
                except Exception as exc:
                    path = future_to_path[future]
                    error_message = f"Error processing '{path}': {exc}"
                    logging.error(error_message)
                    tqdm.write(click.style(f"\n{error_message}", fg="red"), file=sys.stderr)

        results_queue.put(worker.sentinel)
        writer_thread.join()

    def refine_unknown_files(self):
        """
        Finds files with generic MIME types in the database and re-processes them
        using Fido for better identification.
        """
        self._prepare_settings()
    
        if not self.app_config.use_fido:
            click.echo(click.style("Fido is not enabled. Please set 'use_fido = true' in your walker.toml.", fg="yellow"))
            return
    
        click.echo("Querying database for files with unknown MIME types...")
        chunk_size = 10000
        total_refined = 0
    
        with database.get_session() as db_session:
            query = (
                db_session.query(models.FileIndex)
                .filter(models.FileIndex.mime_type.in_(("application/octet-stream", "inode/x-empty")))
            )
            total_to_refine = query.count()
    
            if total_to_refine == 0:
                click.echo("No files with unknown MIME types found to refine.")
                return
    
            click.echo(f"Found {total_to_refine} files to refine with Fido. Processing in chunks...")
            for i in range(0, total_to_refine, chunk_size):
                chunk = query.offset(i).limit(chunk_size).all()
                paths_to_process = [Path(f.path) for f in chunk if Path(f.path).exists()]
                if paths_to_process:
                    self._execute_processing_pool(paths_to_process, f"Refining chunk {i//chunk_size + 1}")
                    total_refined += len(paths_to_process)
    
        click.echo(f"File refinement process complete. {total_refined} files were re-processed.")

    def refine_text_content(self):
        """
        Finds text-based files without content in the database and processes them
        to extract text, generate embeddings, and find PII.
        """
        self._prepare_settings()
        # Force text extraction on for this operation, regardless of config.
        self.app_config.extract_text_on_scan = True

        click.echo("Querying database for text files missing content...")
        chunk_size = 10000
        total_refined = 0

        # MIME types that are likely to contain extractable text.
        text_mime_types = [
            "text/plain", "text/html", "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.oasis.opendocument.text",
            "application/vnd.oasis.opendocument.spreadsheet",
            "application/vnd.oasis.opendocument.presentation",
            "message/rfc822",
        ]

        with database.get_session() as db_session:
            query = (
                db_session.query(models.FileIndex)
                .filter(models.FileIndex.mime_type.in_(text_mime_types))
                .filter(models.FileIndex.content.is_(None))
            )
            total_to_refine = query.count()

            if total_to_refine == 0:
                click.echo("No text files found that need content extraction.")
                return

            click.echo(f"Found {total_to_refine} files to process for text content. Processing in chunks...")
            for i in range(0, total_to_refine, chunk_size):
                chunk = query.offset(i).limit(chunk_size).all()
                paths_to_process = [Path(f.path) for f in chunk if Path(f.path).exists()]
                if paths_to_process:
                    self._execute_processing_pool(paths_to_process, f"Extracting text in chunk {i//chunk_size + 1}")
                    total_refined += len(paths_to_process)

        click.echo(f"Text refinement process complete. {total_refined} files were re-processed.")

    def run(self):
        """
        Executes the main file indexing workflow.
        """
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

        postfix_data = {"processed": 0}
        with tqdm(desc="Scanning for files...", unit=" files", postfix=postfix_data) as pbar:
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

                if files_to_process:
                    self._execute_processing_pool(files_to_process, "Processing chunk")
                    postfix_data["processed"] += len(files_to_process)
                    pbar.set_postfix(postfix_data)

        click.echo("All files have been processed and indexed.")