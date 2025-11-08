# wanderer/reporter.py
from collections import defaultdict
from itertools import groupby
from typing import Optional

import click
import imagehash
import numpy as np
from sqlalchemy import func
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from . import database, models
from .main import format_bytes
from .utils import group_pairs


class Reporter:
    """Handles querying the database and reporting results."""

    def find_dupes(self, print_output: bool = True):
        """Finds files with identical content based on their SHA-256 hash."""
        with database.get_session() as db_session:
            click.echo("Querying for duplicate files by hash...")

            dupe_hashes_subq = (
                db_session.query(models.FileIndex.crypto_hash)
                .group_by(models.FileIndex.crypto_hash)
                .having(func.count(models.FileIndex.id) > 1)
                .subquery()
            )

            all_dupes = (
                db_session.query(models.FileIndex)
                .filter(models.FileIndex.crypto_hash.in_(dupe_hashes_subq))
                .order_by(models.FileIndex.crypto_hash, models.FileIndex.mtime)
                .all()
            )

            if not all_dupes:
                click.echo("No duplicate files found.")
                return []

            grouped_dupes = groupby(all_dupes, key=lambda file: file.crypto_hash)
            results = []
            for hash_val, files_group in grouped_dupes:
                results.append((hash_val, list(files_group)))

            if print_output:
                for i, (hash_val, files) in enumerate(results, 1):
                    click.echo(f"\n--- Set {i} ({len(files)} files, hash: {hash_val[:12]}...) ---")
                    click.echo(click.style(f"  Source: {files[0].path}", fg="green"))
                    for file in files[1:]:
                        click.echo(f"  - Dup:  {file.path}")
            
            return results

    def find_image_dupes(self, threshold: int, print_output: bool = True):
        """Finds visually identical or similar images based on perceptual hash."""
        with database.get_session() as db_session:
            click.echo("Querying for duplicate images by perceptual hash...")

            images = (
                db_session.query(models.FileIndex.path, models.FileIndex.perceptual_hash)
                .filter(models.FileIndex.perceptual_hash.isnot(None))
                .all()
            )

            if len(images) < 2:
                click.echo("Not enough images in the index to compare.")
                return []

            click.echo(f"Comparing {len(images)} images...")
            hashes = {path: imagehash.hex_to_hash(phash) for path, phash in images}
            paths = list(hashes.keys())

            groups = []
            if threshold == 0:
                hash_groups = defaultdict(list)
                for path, phash_str in images:
                    hash_groups[phash_str].append(path)
                groups = [sorted(g) for g in hash_groups.values() if len(g) > 1]
            else:
                # The original approach of creating a full N*N distance matrix is not scalable.
                # It will consume huge amounts of memory (e.g., ~476 GiB for 90k images).
                # We will switch to a more memory-efficient, chunked approach.
                hash_array = np.array([h.hash.flatten() for h in hashes.values()], dtype=bool)
                num_images = len(paths)
                chunk_size = 1000  # Process 1000 images at a time
                similar_pairs = []

                with tqdm(total=(num_images // chunk_size)**2 // 2, desc="Comparing chunks") as pbar:
                    for i in range(0, num_images, chunk_size):
                        chunk_i = hash_array[i:i + chunk_size]
                        
                        # Compare within the chunk
                        dist_matrix_intra = np.sum(chunk_i[:, np.newaxis, :] != chunk_i[np.newaxis, :, :], axis=2)
                        indices_i, indices_j = np.where((dist_matrix_intra <= threshold) & (np.triu(np.ones_like(dist_matrix_intra), k=1) == 1))
                        for i1, j1 in zip(indices_i, indices_j):
                            similar_pairs.append((i + i1, i + j1))

                        # Compare with subsequent chunks
                        for j in range(i + chunk_size, num_images, chunk_size):
                            chunk_j = hash_array[j:j + chunk_size]
                            dist_matrix_inter = np.sum(chunk_i[:, np.newaxis, :] != chunk_j[np.newaxis, :, :], axis=2)
                            indices_i, indices_j = np.where(dist_matrix_inter <= threshold)
                            for i1, j1 in zip(indices_i, indices_j):
                                similar_pairs.append((i + i1, j + j1))
                            pbar.update(1)
                groups = [sorted(g) for g in group_pairs(similar_pairs, paths)]

            if not groups:
                if print_output:
                    click.echo("No similar images found with the given threshold.")
                return []

            if print_output:
                for i, group in enumerate(groups, 1):
                    click.echo(f"\n--- Similar Group {i} ---")
                    for path in group:
                        click.echo(f"  - {path}")
            return groups

    def find_similar_text(self, threshold: float, print_output: bool = True, progress_callback: Optional[callable] = None):
        """Finds files with similar text content using vector embeddings."""
        with database.get_session() as db_session:
            click.echo("Querying for files with text embeddings...")
            query = db_session.query(models.FileIndex.path, models.FileIndex.content_embedding).filter(
                models.FileIndex.content_embedding.isnot(None))

            count = query.count()
            if count < 2:
                click.echo("Not enough text files in the index to compare.")
                return []

            click.echo(f"Found {count} text files. Calculating similarities...")

            # This is a memory-intensive operation. We'll process in chunks to avoid OOM errors.
            chunk_size = 2000  # Adjust based on typical embedding size and available RAM
            similar_pairs = []
            
            # Pre-fetch all data to avoid repeated DB queries inside the loop.
            results = query.all()
            paths = [r.path for r in results]
            embeddings = np.array([np.frombuffer(r.content_embedding, dtype=np.float32) for r in results])

            total_chunks = (count + chunk_size - 1) // chunk_size
            total_comparisons = total_chunks * (total_chunks + 1) // 2
            with tqdm(total=total_comparisons, desc="Comparing chunks", disable=not print_output) as pbar:
                for i in range(0, count, chunk_size):
                    chunk_i_embeddings = embeddings[i:i + chunk_size]

                    # Compare within the chunk (upper triangle)
                    sim_matrix_intra = cosine_similarity(chunk_i_embeddings)
                    indices_i, indices_j = np.where(np.triu(sim_matrix_intra, k=1) >= threshold)
                    for i1, j1 in zip(indices_i, indices_j):
                        similar_pairs.append((i + i1, i + j1))

                    # Compare chunk_i with all subsequent chunks
                    for j in range(i + chunk_size, count, chunk_size):
                        chunk_j_embeddings = embeddings[j:j + chunk_size]
                        sim_matrix_inter = cosine_similarity(chunk_i_embeddings, chunk_j_embeddings)
                        indices_i, indices_j = np.where(sim_matrix_inter >= threshold)
                        for i1, j1 in zip(indices_i, indices_j):
                            similar_pairs.append((i + i1, j + j1))
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(pbar.n, pbar.total, f"Comparing chunk {i//chunk_size + 1}...")

            if not similar_pairs:
                if print_output: click.echo("No similar text files found above the threshold.")
                return []

            groups = group_pairs(similar_pairs, paths)
            if print_output:
                for i, group in enumerate(groups, 1):
                    click.echo(f"\n--- Similar Group {i} ---")
                    for path in sorted(group):
                        click.echo(f"  - {path}")
            return groups

    def search_content(self, full_query: str, limit: int):
        """Performs a semantic search for files based on text content."""
        from .file_processor import get_embedding_model

        with database.get_session() as db_session:
            click.echo(f"Searching for files with content similar to: '{full_query}'")
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.encode([full_query])

            query = db_session.query(models.FileIndex.path, models.FileIndex.content_embedding).filter(
                models.FileIndex.content_embedding.isnot(None))

            if query.count() == 0:
                click.echo("No text files with embeddings found in the index.")
                return

            # Process in chunks to keep memory usage low
            chunk_size = 5000
            all_results = []

            for i in tqdm(range(0, query.count(), chunk_size), desc="Searching"):
                chunk = query.offset(i).limit(chunk_size).all()
                chunk_paths = [r.path for r in chunk]
                chunk_embeddings = np.array([np.frombuffer(r.content_embedding, dtype=np.float32) for r in chunk])
                
                similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                for j, score in enumerate(similarities):
                    all_results.append((score, chunk_paths[j]))

            # Sort all collected results by score and get the top N
            all_results.sort(key=lambda x: x[0], reverse=True)

            click.echo(f"\n--- Top {limit} results ---")
            for score, path in all_results[:limit]:
                click.echo(f"Score: {score:.4f} | {path}")

    def find_lonely_files(self, limit: int):
        """Finds unique files that have no duplicates."""
        with database.get_session() as db_session:
            click.echo("Querying for unique files (no content duplicates)...")

            # Subquery to find all hashes that appear more than once
            dupe_hashes_subq = (
                db_session.query(models.FileIndex.crypto_hash)
                .group_by(models.FileIndex.crypto_hash)
                .having(func.count(models.FileIndex.id) > 1)
                .subquery()
            )

            # Query for files whose hash is NOT in the list of duplicate hashes
            lonely_files_query = (
                db_session.query(models.FileIndex)
                .filter(models.FileIndex.crypto_hash.notin_(dupe_hashes_subq))
                .order_by(models.FileIndex.size_bytes.desc())
            )

            total_lonely = lonely_files_query.count()
            click.echo(f"Found {total_lonely} unique files. Showing the largest {limit}:")

            for file in lonely_files_query.limit(limit).all():
                click.echo(f"{format_bytes(file.size_bytes):>10} | {file.path}")

    def largest_files(self, limit: int, print_output: bool = True) -> list[models.FileIndex]:
        """Lists the largest files in the index by size."""
        with database.get_session() as db_session:
            if print_output:
                click.echo(f"Querying for the {limit} largest files...")

            files = (
                db_session.query(models.FileIndex)
                .order_by(models.FileIndex.size_bytes.desc())
                .limit(limit)
                .all()
            )
            if print_output:
                for file in files:
                    click.echo(f"{format_bytes(file.size_bytes):>10} | {file.path}")

            if not files and print_output:
                click.echo("No files found in the index.")

            return files

    def pronom_summary(self, print_output: bool = True):
        """Shows a summary of file counts grouped by their PRONOM ID."""
        with database.get_session() as db_session:
            click.echo("Generating file type summary by PRONOM ID...")
            summary = (
                db_session.query(
                    models.FileIndex.pronom_id,
                    func.count(models.FileIndex.id).label("count"),
                    func.sum(models.FileIndex.size_bytes).label("total_size")
                )
                .filter(models.FileIndex.pronom_id.isnot(None))
                .group_by(models.FileIndex.pronom_id)
                .order_by(func.count(models.FileIndex.id).desc())
                .all()
            )
            if not summary:
                if print_output:
                    click.echo("No files with PRONOM IDs found in the index. (Run with 'use_fido = true')")
                return []
            if print_output:
                click.echo(f"{'PRONOM ID':<15} | {'Count':>10} | {'Total Size':>12}")
                click.echo("-" * 43)
                for pronom_id, count, total_size in summary:
                    click.echo(f"{str(pronom_id):<15} | {count:>10} | {format_bytes(total_size or 0):>12}")
            return summary

    def type_summary(self, print_output: bool = True):
        """Shows a summary of file counts and sizes grouped by MIME type."""
        with database.get_session() as db_session:
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
                if print_output:
                    click.echo("No files found in the index.")
                return []
            if print_output:
                click.echo(f"{'MIME Type':<60} | {'Count':>10} | {'Total Size':>12}")
                click.echo("-" * 86)
                for mime_type, count, total_size in summary:
                    click.echo(f"{str(mime_type):<60} | {count:>10} | {format_bytes(total_size or 0):>12}")
            return summary

    def list_pii_files(self, print_output: bool = True):
        """Lists all files that have been flagged for containing potential PII."""
        with database.get_session() as db_session:
            click.echo("Querying for files flagged with potential PII...")
            files = (
                db_session.query(models.FileIndex)
                .filter(models.FileIndex.pii_types.isnot(None))
                .order_by(models.FileIndex.path)
                .all()
            )
            if not files:
                if print_output:
                    click.echo("No files containing potential PII were found.")
                return []
            
            if print_output:
                for file in files:
                    pii_types_str = ", ".join(file.pii_types)
                    click.echo(f"{file.path} [{click.style(pii_types_str, fg='yellow')}]")
            return files