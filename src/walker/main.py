# walker/main.py
import logging
import sys
import os, sys
from pathlib import Path
from typing import Optional, Tuple

import click
import imagehash
from sqlalchemy import func
from sqlalchemy.orm import Session
from tqdm import tqdm
import numpy as np

from . import config, database, models, scanner, indexer

def setup_logging():
    """Sets up logging to a file for warnings and errors."""
    log_file = Path(__file__).parent / "walker.log"
    # Configure logging to write to a file, appending to it if it exists.
    # Only messages of level WARNING and above will be logged.
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='a')

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
@click.option('--exclude', 'exclude_paths', multiple=True, type=click.Path(), help='Directory name to exclude. Can be used multiple times.')
def index(root_paths: Tuple[Path, ...], workers: int, exclude_paths: Tuple[str, ...]):
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

    # --- Merge CLI arguments and config file settings ---
    # CLI options take precedence over the config file.
    final_workers = workers if workers != 3 else app_config.workers

    # The click.echo function will be used as the progress callback for the CLI
    indexer_instance = indexer.Indexer(
        root_paths=root_paths,
        workers=final_workers,
        exclude_paths=exclude_paths,
        progress_callback=click.echo,
        progress_bar_callback=tqdm,
    )
    indexer_instance.run()

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
            files = db_session.query(models.FileIndex).filter_by(crypto_hash=hash_val).order_by(models.FileIndex.mtime).all()
            
            # The first file after sorting by mtime is the "source"
            source_file = files[0]
            click.echo(click.style(f"  Source: {source_file.path}", fg="green"))

            # List the other duplicates
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

        # --- Grouping similar images ---
        groups = []
        processed_indices = set()

        for i in range(len(paths)):
            if i in processed_indices:
                continue

            current_group = {paths[i]}
            processed_indices.add(i)

            for j in range(i + 1, len(paths)):
                if j in processed_indices:
                    continue
                
                # Compare perceptual hashes
                if hashes[paths[i]] - hashes[paths[j]] <= threshold:
                    current_group.add(paths[j])
                    processed_indices.add(j)
            
            if len(current_group) > 1:
                groups.append(sorted(list(current_group)))

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

        # --- Grouping using a Disjoint Set Union (DSU) data structure ---
        parent = list(range(len(paths)))
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        for i, j in similar_pairs:
            union(i, j)

        groups = {}
        for i in range(len(paths)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(paths[i])

        for i, group_paths in enumerate(groups.values(), 1):
            if len(group_paths) > 1:
                click.echo(f"\n--- Similar Group {i} ---")
                for path in sorted(group_paths):
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