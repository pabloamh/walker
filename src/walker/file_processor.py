# walker/file_processor.py
import email
import hashlib
import json
import os
import tarfile
import zipfile
from typing import Optional, Tuple, Any, Union

import imagehash
from bs4 import BeautifulSoup
import magic
import pymupdf
from pymediainfo import MediaInfo
from tinytag import TinyTag
from sentence_transformers import SentenceTransformer
from presidio_analyzer import AnalyzerEngine
from PIL import Image, ExifTags
from docx import Document

from pathlib import Path
from .models import FileMetadata

# --- Model Loading ---
# Load the model once when the module is imported. This is crucial for performance,
# as it prevents reloading the model for every file in a multi-processing environment.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Initialize the PII analyzer engine. This is also loaded once for performance.
# It will use a default spaCy model ('en_core_web_lg') if available.
pii_analyzer = AnalyzerEngine()


# Default filenames and extensions to exclude from processing.
# These are checked case-insensitively.
DEFAULT_EXCLUDED_FILENAMES = {
    ".ds_store",
    "thumbs.db",
}
DEFAULT_EXCLUDED_EXTENSIONS = {
    ".swp",  # Vim swap file
    ".swo",  # Vim swap file
    ".tmp",  # Temporary file
    ".part", # Partial download
    ".crdownload", # Chrome partial download
}

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

    def _process_video(self) -> Optional[str]:
        """Extracts metadata from a video file."""
        try:
            media_info = MediaInfo.parse(self.file_path)
            video_track = next((t for t in media_info.tracks if t.track_type == 'Video'), None)
            audio_track = next((t for t in media_info.tracks if t.track_type == 'Audio'), None)

            if not video_track:
                return None

            video_data = {
                "width": video_track.width,
                "height": video_track.height,
                "duration_ms": video_track.duration,
                "frame_rate": video_track.frame_rate,
                "codec": video_track.codec_id,
                "format": video_track.format,
                "audio_codec": audio_track.format if audio_track else None,
            }
            return json.dumps(video_data, default=str)
        except Exception:
            return None

    def _process_audio(self) -> Optional[str]:
        """Extracts metadata from an audio file."""
        try:
            tag = TinyTag.get(self.file_path)
            audio_data = {
                "artist": tag.artist,
                "album": tag.album,
                "album_artist": tag.albumartist,
                "title": tag.title,
                "track": tag.track,
                "year": tag.year,
                "duration_s": tag.duration,
                "genre": tag.genre,
                "bitrate_kbps": tag.bitrate,
            }
            return json.dumps({k: v for k, v in audio_data.items() if v is not None})
        except Exception:
            return None

    def _process_archive(self) -> Optional[str]:
        """Extracts the list of files from a compressed archive."""
        try:
            file_list = []
            if self.mime_type == "application/zip":
                with zipfile.ZipFile(self.file_path, 'r') as zf:
                    file_list = zf.namelist()
            elif tarfile.is_tarfile(self.file_path):
                with tarfile.open(self.file_path, 'r:*') as tf:
                    file_list = tf.getnames()
            
            if file_list:
                archive_data = {"files": file_list, "file_count": len(file_list)}
                return json.dumps(archive_data)
            return None
        except (zipfile.BadZipFile, tarfile.TarError):
            return None # Not a valid archive or corrupted

    def _extract_text_content(self) -> Optional[str]:
        """Extracts text content from various file types based on MIME type."""
        try:
            if not self.mime_type:
                return None

            content = ""

            # Plain text files (and source code, markdown, etc.)
            if self.mime_type.startswith("text/plain"):
                # Read with UTF-8, ignore errors for robustness against malformed files
                content = self.file_path.read_text(encoding="utf-8", errors="ignore")

            # HTML files
            elif self.mime_type == "text/html":
                with self.file_path.open("r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f, "lxml")
                    content = soup.get_text(separator="\n", strip=True)

            # PDF documents
            elif self.mime_type == "application/pdf":
                with pymupdf.open(self.file_path) as doc:
                    for page in doc:
                        content += page.get_text()

            # Microsoft Word documents
            elif self.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(self.file_path)
                for para in doc.paragraphs:
                    content += para.text + "\n"

            # Email messages
            elif self.mime_type == "message/rfc822":
                with self.file_path.open("rb") as f:
                    msg = email.message_from_bytes(f.read())
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                # Decode payload, ignoring errors
                                content += part.get_payload(decode=True).decode(errors="ignore")
                    elif msg.get_content_type() == "text/plain":
                        content = msg.get_payload(decode=True).decode(errors="ignore")

            return content.strip() if content else None
        except Exception:
            return None

    def process(self) -> Optional[FileMetadata]:
        """
        Processes the file and returns a FileMetadata object.
        """
        try:
            # --- Early exit for excluded file types ---
            if self.file_path.name.lower() in DEFAULT_EXCLUDED_FILENAMES:
                return None
            if self.file_path.suffix.lower() in DEFAULT_EXCLUDED_EXTENSIONS:
                return None


            if not self.file_path.is_file():
                return None

            self.mime_type = magic.from_file(str(self.file_path), mime=True)
            
            metadata_kwargs = {
                "path": str(self.file_path),
                "filename": self.file_path.name,
                "size_bytes": (stat_result := self.file_path.stat()).st_size,
                "mtime": stat_result.st_mtime,
                "crypto_hash": self._get_crypto_hash(),
                "mime_type": self.mime_type,
                "perceptual_hash": None,
                "content": None,
                "exif_data": None,
                "content_embedding": None,
                "has_pii": None,
            }

            if self.mime_type:
                if self.mime_type.startswith("image"):
                    p_hash, exif = self._process_image()
                    metadata_kwargs["perceptual_hash"] = p_hash
                    metadata_kwargs["exif_data"] = exif
                elif self.mime_type.startswith("video"):
                    # For videos, we store media metadata in the 'exif_data' field.
                    metadata_kwargs["exif_data"] = self._process_video()
                elif self.mime_type.startswith("audio"):
                    # For audio, we also store media metadata in the 'exif_data' field.
                    metadata_kwargs["exif_data"] = self._process_audio()
                elif self.mime_type in ("application/zip", "application/x-tar", "application/gzip", "application/x-bzip2", "application/x-xz"):
                    # For archives, we store the file list in the 'exif_data' field.
                    metadata_kwargs["exif_data"] = self._process_archive()

                
            # For any text-based format, try to extract content.
            metadata_kwargs["content"] = self._extract_text_content()

            # If content was extracted, generate an embedding for it.
            if metadata_kwargs["content"]:
                embedding = embedding_model.encode(metadata_kwargs["content"])
                metadata_kwargs["content_embedding"] = embedding.tobytes()
                
                # Analyze content for PII
                pii_results = pii_analyzer.analyze(text=metadata_kwargs["content"], language='en')
                metadata_kwargs["has_pii"] = bool(pii_results)


            return FileMetadata(**metadata_kwargs)
        except (IOError, PermissionError) as e:
            # Suppressing print statements in workers for cleaner output
            return None