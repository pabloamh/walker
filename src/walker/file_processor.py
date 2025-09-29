# walker/file_processor.py
import hashlib
import json
import os
from typing import Optional, Tuple, Any, Union

import imagehash
import magic
import pymupdf
from PIL import Image, ExifTags
from docx import Document

from pathlib import Path
from .models import FileMetadata

class FileProcessor:
    """
    Encapsulates the logic for processing a single file to extract metadata.
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.mime_type: Optional[str] = None

    def _get_crypto_hash(self) -> str:
        """Calculates the SHA-256 hash of the file."""
        sha256_hash = hashlib.sha256()
        with self.file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _process_exif(self, img: Image.Image) -> Optional[str]:
        """Extracts and serializes EXIF data from an image."""
        try:
            exif_data = img.getexif()
            if not exif_data:
                return None

            # Decode EXIF tags and handle non-serializable data
            decoded_exif: dict[str, Any] = {}
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                # Decode bytes to string if possible, otherwise skip
                if isinstance(value, bytes):
                    try:
                        decoded_exif[str(tag_name)] = value.decode('utf-8', errors='ignore')
                    except UnicodeDecodeError:
                        continue
                else:
                    decoded_exif[str(tag_name)] = value
            
            return json.dumps(decoded_exif, default=str)
        except Exception:
            return None

    def _process_image(self) -> Tuple[Optional[str], Optional[str]]:
        """Generates a perceptual hash and extracts EXIF data for an image file."""
        try:
            with Image.open(self.file_path) as img:
                p_hash = str(imagehash.phash(img))
                exif = self._process_exif(img)
                return p_hash, exif
        except Exception:
            return None, None

    def _process_document(self) -> Optional[str]:
        """Extracts text content from a document."""
        content = ""
        try:
            if self.mime_type and "pdf" in self.mime_type:
                with pymupdf.open(self.file_path) as doc:
                    for page in doc:
                        content += page.get_text()
            elif self.mime_type and "openxmlformats-officedocument.wordprocessingml.document" in self.mime_type:
                doc = Document(self.file_path)
                for para in doc.paragraphs:
                    content += para.text + "\n"
            return content.strip() if content else None
        except Exception:
            return None

    def process(self) -> Optional[FileMetadata]:
        """
        Processes the file and returns a FileMetadata object.
        """
        try:
            if not self.file_path.is_file():
                return None

            self.mime_type = magic.from_file(str(self.file_path), mime=True)
            
            metadata_kwargs = {
                "path": str(self.file_path),
                "filename": self.file_path.name,
                "size_bytes": self.file_path.stat().st_size,
                "crypto_hash": self._get_crypto_hash(),
                "mime_type": self.mime_type,
                "perceptual_hash": None,
                "content": None,
                "exif_data": None,
            }

            if self.mime_type:
                if self.mime_type.startswith("image"):
                    p_hash, exif = self._process_image()
                    metadata_kwargs["perceptual_hash"] = p_hash
                    metadata_kwargs["exif_data"] = exif
                elif self.mime_type.startswith("application"):
                    metadata_kwargs["content"] = self._process_document()

            return FileMetadata(**metadata_kwargs)
        except (IOError, PermissionError) as e:
            print(f"Could not process {self.file_path}: {e}")
            return None