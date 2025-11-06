# wanderer/worker.py
import logging
import queue
import resource
import sys
import warnings
from pathlib import Path
from typing import Optional

import attrs
import click
from PIL import Image
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from . import database, models
from .models import FileMetadata

sentinel = "DONE"  # A signal to stop the writer thread.


def db_writer_worker(db_queue: queue.Queue, batch_size: int):
    """
    A dedicated worker that pulls results from the queue and writes them to the DB.
    """
    def commit_batch(session: Session, current_batch: list):
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
                if "database is locked" in str(e).lower():
                    click.echo(click.style("\nDatabase is locked. Please close other connections.", fg="red"), err=True)
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
            if item == sentinel:
                db_queue.task_done()
                # Final commit for any remaining items in the batch
                total_processed += commit_batch(db_session, batch)
                # All sentinels are in, wait for all tasks to be marked as done.
                db_queue.join()
                db_queue.task_done()
                break
            else:
                batch.append(attrs.asdict(item)) # type: ignore
                if len(batch) >= batch_size:
                    total_processed += commit_batch(db_session, batch)
                db_queue.task_done()
    click.echo(f"DB writer finished. A total of {total_processed} records were written.")


def set_memory_limit(limit_gb: Optional[float]):
    if limit_gb is None or sys.platform == "win32":
        return
    try:
        limit_bytes = int(limit_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, resource.error) as e:
        logging.warning(f"Could not set memory limit: {e}")


def process_file_wrapper(path: Path, app_config: models.Config, shared_queue: queue.Queue, memory_limit_gb: Optional[float]) -> bool:
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    set_memory_limit(memory_limit_gb)
    from .file_processor import FileProcessor
    processor = FileProcessor(path, app_config=app_config)
    count = 0
    for metadata in processor.process():
        shared_queue.put(metadata)
        count += 1
    return count > 0