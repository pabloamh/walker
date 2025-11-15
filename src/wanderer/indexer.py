# wanderer/indexer.py
import logging
import multiprocessing
from datetime import datetime
import queue
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple
import os
import click
import attrs
from sqlalchemy.orm import Session
from tqdm import tqdm

from . import config, database, models, scanner
from .models import Config
from . import worker, file_processor


class Indexer:
    """Orchestrates the file scanning, processing, and indexing workflow."""

    def __init__(
        self,
        root_paths: Tuple[Path, ...],
        workers: int,
        memory_limit_gb: Optional[float],
        exclude_paths: Tuple[str, ...],
        app_config: Optional[Config] = None,
        progress_callback: Optional[callable] = None,
    ):
        """
        Initializes the Indexer with CLI arguments and application configuration.

        Args:
            root_paths: A tuple of root directories to start scanning from.
            workers: The number of worker processes to use.
            memory_limit_gb: An optional soft memory limit for each worker in GB.
            exclude_paths: A tuple of paths or patterns to exclude from the scan.
            progress_callback: An optional callable to report progress to a GUI.
            app_config: An optional Config object. If not provided, it's loaded from file.
        """
        self.app_config = app_config or config.load_config()
        self.cli_root_paths = root_paths
        self.cli_workers = workers
        self.cli_memory_limit_gb = memory_limit_gb
        self.cli_exclude_paths = exclude_paths

        self.final_root_paths: Tuple[Path, ...] = tuple()
        self.final_workers: int = 0
        self.final_memory_limit: Optional[float] = None
        self.final_exclude_list: set[str] = set()
        self.progress_callback = progress_callback

    def _prepare_paths_and_exclusions(self):
        """
        Validates root paths and merges exclusion lists from the CLI and config file.
        Also applies platform-specific default exclusions for root-level scans.
        """
        # --- Determine which paths to scan ---
        combined_paths = list(self.cli_root_paths)
        if not combined_paths and self.app_config.scan_dirs:
            click.echo("Using 'scan_dirs' from wanderer.toml.")
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

        # --- Apply platform-specific default exclusions based on scan content, not host OS ---
        for root_path in self.final_root_paths:
            # Check if the root path looks like a Windows system drive
            if any((root_path / p).exists() for p in ["Windows", "pagefile.sys", "Program Files"]):
                click.echo(click.style(f"Windows system-like directory detected at '{root_path}'. Applying default Windows exclusions.", fg="blue"))
                for exclude in models.DEFAULT_WINDOWS_EXCLUDES:
                    self.final_exclude_list.add(os.path.normcase(str(root_path / exclude)))

            # Check if the root path looks like a macOS system drive
            elif any((root_path / p).exists() for p in ["System", "Library", "Applications"]):
                click.echo(click.style(f"macOS system-like directory detected at '{root_path}'. Applying default macOS exclusions.", fg="blue"))
                for exclude in models.DEFAULT_MACOS_EXCLUDES:
                    # These are often defined from root, so we construct the full path
                    self.final_exclude_list.add(os.path.normcase(str(root_path / exclude.lstrip('/'))))

            # Check for scanning the root of a Linux-like system
            elif root_path.parent == root_path: # This checks if path is '/'
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

    def _filter_chunk(self, path_chunk: list[Path], db_session: Session) -> list[Path]:
        """
        Filters a chunk of paths to find new or modified files.

        This compares files against the database in a memory-efficient way.

        Args:
            path_chunk: A list of file paths to filter.
            db_session: The SQLAlchemy session to use for querying.

        Returns:
            A list of paths for files that need to be processed.
        """
        files_to_process = []
        # Create a map of path strings to Path objects for quick lookup
        path_map = {str(p.resolve()): p for p in path_chunk}
        
        # Query the DB for existing files that are in the current chunk
        existing_files_from_db = db_session.query(models.FileIndex.path, models.FileIndex.mtime).filter(models.FileIndex.path.in_(path_map.keys())).all()
        existing_files_map = {path: mtime for path, mtime in existing_files_from_db}

        for p in path_chunk:
            try:
                path_str = str(p.resolve())
                mtime = p.stat().st_mtime
                if path_str not in existing_files_map or mtime > existing_files_map[path_str]:
                    files_to_process.append(p)
            except FileNotFoundError:
                # File was deleted between scanning and filtering
                continue
        return files_to_process

    def _execute_processing_pool(self, paths_to_process: list[Path], description: str) -> None:
        """
        Manages a process pool to process a list of files and write results to the DB.

        Args:
            paths_to_process: A list of file paths to be processed by the workers.
            description: A description for the tqdm progress bar.
        """
        # A Manager is required to create a Queue that can be shared between processes
        # created by the ProcessPoolExecutor. A standard queue.Queue will not work.
        manager = multiprocessing.Manager()
        results_queue = manager.Queue()

        # Use a threading.Event to signal when the writer is truly finished.
        writer_finished_event = threading.Event()
        writer_thread = threading.Thread(target=worker.db_writer_worker, args=(results_queue, self.app_config.db_batch_size, self.app_config, writer_finished_event))
        writer_thread.start()

        with ProcessPoolExecutor(max_workers=self.final_workers) as executor:
            # Submit all tasks to the process pool, passing the process-safe queue.
            future_to_path = {
                executor.submit(worker.process_file_wrapper, path, self.app_config, results_queue, self.final_memory_limit): path
                for path in paths_to_process
            }

            # Use tqdm for CLI progress, or just iterate if a GUI callback is provided
            iterator = as_completed(future_to_path)
            if not self.progress_callback:
                iterator = tqdm(iterator, total=len(future_to_path), desc=description)

            for i, future in enumerate(iterator):
                if self.progress_callback:
                    self.progress_callback(i + 1, len(future_to_path), description)
                try:
                    future.result()  # Raise exceptions from workers if any
                except Exception as exc:
                    path = future_to_path[future]
                    error_message = f"Error processing '{path}': {exc}"
                    logging.error(error_message, exc_info=True)
                    if not self.progress_callback:
                        tqdm.write(click.style(f"\n{error_message}", fg="red"), file=sys.stderr)

        # Wait for the queue to be fully processed before signaling the writer to stop.
        # This prevents a race condition where the sentinel is processed before all items are written.
        results_queue.join()
        results_queue.put(worker.sentinel)  # Signal the writer to finish
        writer_finished_event.wait()  # Wait for the writer to confirm it's done
        writer_thread.join()  # Cleanly join the thread

    def _execute_droid_refinement(self, paths_to_process: list[Path], description: str) -> None:
        """
        Manages a DROID batch process for a list of files, updating the DB directly.
        This is much faster than the one-by-one process pool for DROID.

        Args:
            paths_to_process: A list of file paths to be processed by DROID.
            description: A description for the tqdm progress bar.
        """
        # Group files by their parent directory to run DROID efficiently.
        dirs_to_scan = {p.parent for p in paths_to_process}

        with database.get_session() as db_session:
            iterator = tqdm(dirs_to_scan, desc=description, unit="dir") if not self.progress_callback else dirs_to_scan

            for i, directory in enumerate(iterator):
                if self.progress_callback:
                    self.progress_callback(i + 1, len(dirs_to_scan), f"DROID: {directory.name}")
                elif isinstance(iterator, tqdm):
                    iterator.set_postfix({"dir": str(directory)})

                updates = []
                try:
                    # 1. Run DROID and get results for the current directory.
                    droid_results = list(file_processor.FileProcessor.get_pronom_ids_in_batch(directory, self.app_config))
                    if not droid_results:
                        continue

                    # 2. Extract file paths from DROID results to query the DB.
                    paths_from_droid = [res[0] for res in droid_results]

                    # 3. Fetch the primary keys (id) for the files we need to update.
                    #    This is the crucial step to fix the InvalidRequestError.
                    existing_files = db_session.query(models.FileIndex.id, models.FileIndex.path).filter(models.FileIndex.path.in_(paths_from_droid)).all()
                    path_to_id_map = {path: file_id for file_id, path in existing_files}

                    # 4. Build the list of updates, now including the primary key 'id'.
                    for absolute_path_str, puid, mimetype in droid_results:
                        if absolute_path_str not in path_to_id_map: continue
                        update_data = {"id": path_to_id_map[absolute_path_str], "pronom_id": puid}
                        if mimetype and mimetype and mimetype != "application/octet-stream":
                            update_data["mime_type"] = mimetype
                        updates.append(update_data)

                    if updates:
                        # 5. Perform the bulk update.
                        db_session.bulk_update_mappings(models.FileIndex, updates)
                        db_session.commit()
                except Exception as e:
                    logging.error(f"Error during DROID batch refinement for {directory}: {e}", exc_info=True)
                    db_session.rollback()


    def refine_unknown_files(self):
        """
        Finds all files in the database and re-processes them using DROID
        for better identification.
        """
        self._prepare_settings()
    
        if not self.app_config.use_droid:
            click.echo(click.style("DROID is not enabled. Please set 'use_droid = true' in your wanderer.toml.", fg="yellow"))
            return
    
        click.echo("Querying database for all files to refine with DROID...")
        chunk_size = 10000
        total_refined = 0
    
        with database.get_session() as db_session:
            # Query for all files to do a full DROID refinement.
            query = db_session.query(models.FileIndex)
            total_to_refine = query.count()
    
            if total_to_refine == 0:
                click.echo("No files found in the database to refine.")
                return
    
            click.echo(f"Found {total_to_refine} files to refine with DROID. Processing in chunks...")
            for i in range(0, total_to_refine, chunk_size):
                chunk = query.offset(i).limit(chunk_size).all()
                # Explicitly resolve the path before checking existence to ensure correctness,
                # regardless of the current working directory.
                paths_to_process = [Path(f.path).resolve() for f in chunk if Path(f.path).exists()]
                if paths_to_process:
                    self._execute_droid_refinement(paths_to_process, f"Refining chunk {i//chunk_size + 1}")
    
        click.echo(f"DROID refinement process complete. {total_refined} files were re-processed.")

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
                paths_to_process = [Path(f.path).resolve() for f in chunk if Path(f.path).exists()]
                if paths_to_process:
                    self._execute_processing_pool(paths_to_process, f"Extracting text in chunk {i//chunk_size + 1}")
                    total_refined += len(paths_to_process)

        click.echo(f"Text refinement process complete. {total_refined} files were re-processed.")

    def refine_text_content_by_path(self, paths: Tuple[Path, ...]):
        """
        Finds files under a specific path without content and processes them
        to extract text, generate embeddings, and find PII.
        """
        self._prepare_settings()
        # Force text extraction on for this operation.
        self.app_config.extract_text_on_scan = True

        click.echo(f"Querying database for text files under the specified paths...")
        chunk_size = 10000
        total_refined = 0

        with database.get_session() as db_session:
            # Build a list of LIKE clauses for each path
            path_filters = [models.FileIndex.path.like(f"{str(p.resolve())}%") for p in paths]
            
            query = (
                db_session.query(models.FileIndex)
                .filter(database.or_(*path_filters))
                .filter(models.FileIndex.content.is_(None))
            )
            total_to_refine = query.count()

            if total_to_refine == 0:
                click.echo("No files found under the specified paths that need content extraction.")
                return

            click.echo(f"Found {total_to_refine} files to process for text content. Processing in chunks...")
            for i in range(0, total_to_refine, chunk_size):
                chunk = query.offset(i).limit(chunk_size).all()
                paths_to_process = [Path(f.path).resolve() for f in chunk if Path(f.path).exists()]
                if paths_to_process:
                    self._execute_processing_pool(paths_to_process, f"Extracting text in chunk {i//chunk_size + 1}")
                    total_refined += len(paths_to_process)

        click.echo(f"Path-based text refinement complete. {total_refined} files were re-processed.")

    def refine_image_content_by_path(self, paths: Tuple[Path, ...], mime_types: Optional[Tuple[str, ...]] = None):
        """
        Finds image files under a specific path without a perceptual hash
        and processes them to generate it. Can be filtered by MIME type.
        """
        self._prepare_settings()
        self.app_config = attrs.evolve(self.app_config, compute_perceptual_hash=True)
        # Force perceptual hash computation on for this operation.
        self.app_config.compute_perceptual_hash = True

        click.echo(f"Querying database for image files under the specified paths...")
        chunk_size = 10000
        total_refined = 0

        with database.get_session() as db_session:
            # Build a list of LIKE clauses for each path
            path_filters = [models.FileIndex.path.like(f"{str(p.resolve())}%") for p in paths]

            query = (
                db_session.query(models.FileIndex)
                .filter(database.or_(*path_filters))
                .filter(
                    models.FileIndex.mime_type.in_(mime_types)
                    if mime_types
                    else models.FileIndex.mime_type.like("image/%")
                )
                .filter(models.FileIndex.perceptual_hash.is_(None))
            )
            total_to_refine = query.count()

            if total_to_refine == 0:
                click.echo("No image files found under the specified paths that need p-hash computation.")
                return

            click.echo(f"Found {total_to_refine} images to process for perceptual hashes. Processing in chunks...")
            for i in range(0, total_to_refine, chunk_size):
                chunk = query.offset(i).limit(chunk_size).all()
                paths_to_process = [Path(f.path).resolve() for f in chunk if Path(f.path).exists()]
                if paths_to_process:
                    self._execute_processing_pool(paths_to_process, f"Computing p-hashes in chunk {i//chunk_size + 1}")
                    total_refined += len(paths_to_process)

        click.echo(f"Path-based image refinement complete. {total_refined} files were re-processed.")

    def refine_droid_by_path(self, paths: Tuple[Path, ...]):
        """
        Forces a DROID rescan on all files under a specific path.
        """
        self._prepare_settings()
        if not self.app_config.use_droid:
            click.echo(click.style("DROID is not enabled. Please set 'use_droid = true' in your wanderer.toml.", fg="yellow"))
            return

        click.echo(f"Querying database for all files under the specified paths for DROID rescan...")
        chunk_size = 10000
        total_refined = 0

        with database.get_session() as db_session:
            path_filters = [models.FileIndex.path.like(f"{str(p.resolve())}%") for p in paths]
            
            query = (
                db_session.query(models.FileIndex)
                .filter(database.or_(*path_filters))
            )
            total_to_refine = query.count()

            if total_to_refine == 0:
                click.echo("No files found under the specified paths to refine with DROID.")
                return

            click.echo(f"Found {total_to_refine} files to refine with DROID. Processing in chunks...")
            for i in range(0, total_to_refine, chunk_size):
                chunk = query.offset(i).limit(chunk_size).all()
                paths_to_process = [Path(f.path).resolve() for f in chunk if Path(f.path).exists()]
                if paths_to_process:
                    self._execute_droid_refinement(paths_to_process, f"DROID-refining chunk {i//chunk_size + 1}")

        click.echo(f"Path-based DROID refinement complete.")

    def run(self):
        """
        Executes the main file indexing workflow.
        """
        self._prepare_settings()
        self._prepare_paths_and_exclusions()

        if not self.final_root_paths:
            config_path = Path("wanderer.toml")
            if not config_path.is_file():
                click.echo(click.style("Warning: No scan paths provided and 'wanderer.toml' not found.", fg="yellow"))
                click.echo(f"Please create a '{config_path.resolve()}' file with 'scan_dirs' or specify paths on the command line.")
            else:
                click.echo(click.style("Warning: No scan paths provided.", fg="yellow"))
                click.echo("Please add directories to 'scan_dirs' in your 'wanderer.toml' or specify paths on the command line.")
            return

        # --- Create Scan Log Entry ---
        scan_log_id = None
        with database.get_session() as db_session:
            new_log = models.ScanLog(
                start_time=datetime.now(),
                root_paths=[str(p) for p in self.final_root_paths],
                status='started'
            )
            db_session.add(new_log)
            db_session.commit()
            scan_log_id = new_log.id

        def report_progress(value, total, description):
            if self.progress_callback:
                self.progress_callback(value, total, description)

        total_files_processed = 0
        all_processed_paths = []
        try:
            click.echo(f"Starting scan with {self.final_workers} workers...")
            if self.final_memory_limit and sys.platform != "win32":
                click.echo(click.style(f"Applying a soft memory limit of {self.final_memory_limit:.2f} GB per worker.", fg="blue"))

            with database.get_session() as db_session:
                postfix_data = {"processed": 0}
                with tqdm(desc="Scanning for files...", unit=" files", postfix=postfix_data) as pbar:
                    chunk_iterator = self._get_file_chunks(chunk_size=10000)
                    for path_chunk in chunk_iterator:
                        report_progress(pbar.n, pbar.total, "Filtering files...")
                        files_to_process = self._filter_chunk(path_chunk, db_session)
                        pbar.update(len(path_chunk))

                        if files_to_process:
                            report_progress(pbar.n, pbar.total, f"Processing {len(files_to_process)} new/modified files...")
                            self._execute_processing_pool(files_to_process, "Processing chunk")
                            all_processed_paths.extend(files_to_process)
                            total_files_processed += len(files_to_process)
                            pbar.set_postfix({"processed": total_files_processed})

            # --- Automatic DROID Refinement ---
            # If DROID is enabled, run a batch refinement on all the files that were just processed.
            # This is much more efficient than running DROID one-by-one during the initial scan.
            if self.app_config.use_droid and all_processed_paths:
                click.echo(f"\nDROID is enabled. Running batch analysis on {len(all_processed_paths)} newly processed files...")
                self._execute_droid_refinement(all_processed_paths, "DROID Batch Analysis")
            click.echo("All files have been processed and indexed.")
            status = 'completed'
        except Exception as e:
            status = 'failed'
            logging.error(f"Scan failed: {e}", exc_info=True)
            click.echo(click.style(f"Scan failed with an error: {e}", fg="red"), err=True)
        finally:
            # --- Update Scan Log Entry ---
            if scan_log_id:
                with database.get_session() as db_session:
                    log_entry = db_session.get(models.ScanLog, scan_log_id)
                    if log_entry:
                        log_entry.end_time = datetime.now()
                        log_entry.files_scanned = total_files_processed
                        log_entry.status = status
                        db_session.commit()
                        click.echo(f"Scan log entry '{scan_log_id}' updated with status: {status}")
        
        report_progress(1, 1, f"Scan {status}.")