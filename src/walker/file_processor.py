# walker/file_processor.py
import email
import logging
import warnings
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
from . import config
from .models import FileMetadata, FileIndex

# --- Filter specific warnings ---
# Pillow >= 10.1.0 raises a DecompressionBombWarning for large images.
# We trust the files we are scanning, so we can suppress this.
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --- Model Loading ---
def get_embedding_model() -> SentenceTransformer:
    """Loads the SentenceTransformer model."""
    app_config = config.load_config()
    model_name_or_path = app_config.embedding_model_path or 'all-MiniLM-L6-v2'

    # If a relative path is provided, make it relative to the script's directory
    # to ensure it's found correctly regardless of where the app is run from.
    if app_config.embedding_model_path and not os.path.isabs(app_config.embedding_model_path):
        script_dir = Path(__file__).parent
        model_name_or_path = str(script_dir / app_config.embedding_model_path)

    return SentenceTransformer(model_name_or_path, device='cpu')

# Load models once when the module is imported. This is crucial for performance,
# as it prevents reloading the models for every file in a multi-processing environment.

def get_pii_analyzer() -> AnalyzerEngine:
    """Loads the Presidio AnalyzerEngine with languages from config."""
    app_config = config.load_config()
    # We need to create a new registry and load the NLP models for the specified languages.
    # This is a more advanced setup to support multiple languages.
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    provider = NlpEngineProvider()
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=app_config.pii_languages)

embedding_model = get_embedding_model()
pii_analyzer = get_pii_analyzer()


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
    ".log", # Log files
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

    def _process_text_pii_in_chunks(self, file_path: Path, encoding: str = "utf-8") -> bool:
        """
        Scans a text file for PII in chunks to avoid loading the whole file into memory.
        Returns True as soon as PII is found.
        """
        try:
            app_config = config.load_config()
            primary_language = app_config.pii_languages[0]
            # Presidio's default character limit is 1,000,000. We'll use a smaller chunk size.
            chunk_size = 500_000 

            with file_path.open("r", encoding=encoding, errors="ignore") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    pii_results = pii_analyzer.analyze(text=chunk, language=primary_language)
                    if pii_results:
                        return True # Found PII, no need to scan further
            return False
        except Exception as e:
            logging.warning(f"Error during chunked PII scan for {file_path}: {e}")
            return False

    def _detect_pii(self, content: Optional[str]) -> Optional[bool]:
        """Analyzes extracted content for PII."""
        if not content:
            return None
        try:
            app_config = config.load_config()
            # Truncate content to avoid errors with very large files in presidio.
            truncated_content = content[:1_000_000]
            pii_results = pii_analyzer.analyze(text=truncated_content, language=app_config.pii_languages[0])
            return bool(pii_results)
        except Exception:
            return None

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

            # Open Document Format (ODF) files
            elif self.mime_type in (
                "application/vnd.oasis.opendocument.text",  # .odt
                "application/vnd.oasis.opendocument.spreadsheet",  # .ods
                "application/vnd.oasis.opendocument.presentation",  # .odp
            ):
                from odf import text, teletype
                from odf.opendocument import load as odf_load
                doc = odf_load(self.file_path)
                content = "\n".join(teletype.extractText(p) for p in doc.getElementsByType(text.P))

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

            # Handle cases where the file might be deleted between scanning and processing
            if not self.file_path.exists():
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
            
            # --- PII Detection ---
            # If content was fully extracted, analyze it.
            if metadata_kwargs["content"]:
                metadata_kwargs["has_pii"] = self._detect_pii(metadata_kwargs["content"])
            # For plain text files that might be too large, scan them in chunks without storing content.
            elif self.mime_type and self.mime_type.startswith("text/plain") and metadata_kwargs["size_bytes"] > 1_000_000:
                metadata_kwargs["has_pii"] = self._process_text_pii_in_chunks(self.file_path)

            return FileMetadata(**metadata_kwargs)
        except FileNotFoundError as e:
            # File was likely deleted between the scan and processing.
            logging.warning(f"File not found during processing (likely deleted): {self.file_path} - {e}")
            return None
        except (IOError, PermissionError) as e:
            # Log other I/O related errors, like permission denied.
            logging.warning(f"I/O error processing file: {self.file_path} - {e}")
            return None
        except Exception as e:
            # Catch any other unexpected errors during file processing
            logging.error(f"Unexpected error in FileProcessor for {self.file_path}: {e}", exc_info=True)
            return None