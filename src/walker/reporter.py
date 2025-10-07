# walker/reporter.py
from collections import defaultdict
from itertools import groupby

import click
import imagehash
import numpy as np
from sqlalchemy import func
from sklearn.metrics.pairwise import cosine_similarity

from . import database, models
from .main import format_bytes
from .utils import group_pairs


class Reporter:
    """Handles querying the database and reporting results."""

    def find_dupes(self):
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
                return

            grouped_dupes = groupby(all_dupes, key=lambda file: file.crypto_hash)

            for i, (hash_val, files_group) in enumerate(grouped_dupes, 1):
                files = list(files_group)
                click.echo(f"\n--- Set {i} ({len(files)} files, hash: {hash_val[:12]}...) ---")
                click.echo(click.style(f"  Source: {files[0].path}", fg="green"))
                for file in files[1:]:
                    click.echo(f"  - Dup:  {file.path}")

    def find_image_dupes(self, threshold: int):
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
                return

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
                hash_array = np.array([h.hash.flatten() for h in hashes.values()], dtype=np.uint8)
                diff_matrix = np.not_equal(hash_array[np.newaxis, :, :], hash_array[:, np.newaxis, :])
                distance_matrix = np.sum(diff_matrix, axis=2)
                similar_indices = np.argwhere((distance_matrix <= threshold) & (np.triu(np.ones_like(distance_matrix), k=1) == 1))
                path_groups = group_pairs(similar_indices, paths)
                groups = [sorted(g) for g in path_groups]

            if not groups:
                click.echo("No similar images found with the given threshold.")
                return

            for i, group in enumerate(groups, 1):
                click.echo(f"\n--- Similar Group {i} ---")
                for path in group:
                    click.echo(f"  - {path}")

    def find_similar_text(self, threshold: float):
        """Finds files with similar text content using vector embeddings."""
        with database.get_session() as db_session:
            click.echo("Querying for files with text content...")
            results = db_session.query(models.FileIndex.path, models.FileIndex.content_embedding).filter(
                models.FileIndex.content_embedding.isnot(None)
            ).all()

            if len(results) < 2:
                click.echo("Not enough text files in the index to compare.")
                return

            click.echo(f"Found {len(results)} text files. Calculating similarities...")
            paths = [r.path for r in results]
            embeddings = np.array([np.frombuffer(r.content_embedding, dtype=np.float32) for r in results])
            similarity_matrix = cosine_similarity(embeddings)
            similar_pairs = np.argwhere(np.triu(similarity_matrix, k=1) >= threshold)

            if similar_pairs.shape[0] == 0:
                click.echo("No similar text files found above the threshold.")
                return

            groups = group_pairs(similar_pairs, paths)
            for i, group in enumerate(groups, 1):
                click.echo(f"\n--- Similar Group {i} ---")
                for path in sorted(group):
                    click.echo(f"  - {path}")

    def search_content(self, full_query: str, limit: int):
        """Performs a semantic search for files based on text content."""
        from .file_processor import embedding_model

        with database.get_session() as db_session:
            click.echo(f"Searching for files with content similar to: '{full_query}'")
            query_embedding = embedding_model.encode([full_query])

            results = db_session.query(models.FileIndex.path, models.FileIndex.content_embedding).filter(
                models.FileIndex.content_embedding.isnot(None)
            ).all()

            if not results:
                click.echo("No text files with embeddings found in the index.")
                return

            paths = [r.path for r in results]
            file_embeddings = np.array([np.frombuffer(r.content_embedding, dtype=np.float32) for r in results])

            similarities = cosine_similarity(query_embedding, file_embeddings)[0]
            top_indices = np.argsort(similarities)[-limit:][::-1]

            click.echo(f"\n--- Top {limit} results ---")
            for i in top_indices:
                click.echo(f"Score: {similarities[i]:.4f} | {paths[i]}")

    def largest_files(self, limit: int):
        """Lists the largest files in the index by size."""
        with database.get_session() as db_session:
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

    def type_summary(self):
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
                click.echo("No files found in the index.")
                return
            click.echo(f"{'MIME Type':<60} | {'Count':>10} | {'Total Size':>12}")
            click.echo("-" * 86)
            for mime_type, count, total_size in summary:
                click.echo(f"{str(mime_type):<60} | {count:>10} | {format_bytes(total_size or 0):>12}")

    def list_pii_files(self):
        """Lists all files that have been flagged for containing potential PII."""
        with database.get_session() as db_session:
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