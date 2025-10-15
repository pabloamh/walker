# walker/file_processor.py
import email
import logging
import functools
import warnings
import hashlib
import json
import os
import subprocess
import tarfile
import zipfile
import tempfile
from typing import Optional, Tuple, Any, Union, List, Generator

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
# Pillow also warns about palette images with transparency. This is common
# and doesn't indicate an error for our purposes, so we can suppress it.
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

# Pillow can issue warnings for non-standard TIFF metadata, which is common.
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# --- Model Loading ---
@functools.lru_cache(maxsize=None)
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

def get_spacy_model_name(lang_code: str) -> str:
    """Gets the default spaCy model name for a given language code."""
    # This mapping can be expanded for more languages
    model_map = {
        "en": "en_core_web_lg",
        "es": "es_core_news_md", # Using 'md' as it's smaller and often sufficient
        "fr": "fr_core_news_lg",
    }
    return model_map.get(lang_code, f"{lang_code}_core_news_lg")

@functools.lru_cache(maxsize=None)
def get_pii_analyzer() -> AnalyzerEngine:
    """Loads the Presidio AnalyzerEngine with languages from config."""
    app_config = config.load_config()
    # To prevent noisy warnings about unsupported languages, we will explicitly
    # create a provider configuration that only loads the models and recognizers we need.
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.recognizer_registry import RecognizerRegistry

    provider_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": lang, "model_name": get_spacy_model_name(lang)} for lang in app_config.pii_languages]
    }

    provider = NlpEngineProvider(nlp_configuration=provider_config)
    nlp_engine = provider.create_engine()

    # Create a registry and explicitly load recognizers for the supported languages
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=app_config.pii_languages)

    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=app_config.pii_languages)

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

# Maximum file size for full text extraction to prevent memory issues.
# Files larger than this will not have their content stored in the database.
MAX_CONTENT_EXTRACTION_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

class FileProcessor:
    """
    Encapsulates the logic for processing a single file to extract metadata.
    """
    def __init__(self, file_path: Path, virtual_path: Optional[str] = None, is_archived: bool = False):
        self.file_path = file_path
        self.mime_type: Optional[str] = None
        # The virtual_path is used for files extracted from archives.
        self.virtual_path = virtual_path or str(self.file_path)
        self.app_config = config.load_config()
        self.is_archived = is_archived

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

    def _process_archive(self) -> Generator[FileMetadata, None, None]:
        """
        Extracts files from a compressed archive to a temporary directory
        and yields FileMetadata for each processed file within the archive.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                extracted_files = []

                if self.mime_type == "application/zip":
                    with zipfile.ZipFile(self.file_path, 'r') as zf:
                        zf.extractall(temp_path)
                        extracted_files = [temp_path / f for f in zf.namelist() if (temp_path / f).is_file()]
                elif tarfile.is_tarfile(self.file_path):
                    with tarfile.open(self.file_path, 'r:*') as tf:
                        tf.extractall(temp_path)
                        extracted_files = [temp_path / f for f in tf.getnames() if (temp_path / f).is_file()]

                for extracted_file in extracted_files:
                    # Construct the virtual path as requested.
                    virtual_path = f"{self.virtual_path}/{extracted_file.relative_to(temp_path)}"
                    processor = FileProcessor(extracted_file, virtual_path=virtual_path, is_archived=True)
                    yield from processor.process()

        except (zipfile.BadZipFile, tarfile.TarError, Exception) as e:
            logging.warning(f"Could not process archive {self.file_path}: {e}")
            return # Stop processing this archive if an error occurs

    def _process_text_pii_in_chunks(self, file_path: Path, encoding: str = "utf-8") -> Optional[list[str]]:
        # For very large files, first check if it's likely to be text.
        if "text" not in self.mime_type:
            return None

        """
        Scans a text file for PII in chunks to avoid loading the whole file into memory.
        Returns a list of found PII types.
        """
        try:
            pii_analyzer = get_pii_analyzer()
            found_pii_types: set[str] = set()
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
                        found_pii_types.update(p.entity_type for p in pii_results)
            return sorted(list(found_pii_types)) if found_pii_types else None
        except Exception as e:
            logging.warning(f"Error during chunked PII scan for {file_path}: {e}")
            return None

    def _detect_pii(self, content: Optional[str]) -> Optional[list[str]]:
        """Analyzes extracted content for PII."""
        if not content:
            return None
        try:
            pii_analyzer = get_pii_analyzer()
            app_config = config.load_config()
            # Truncate content to avoid errors with very large files in presidio
            truncated_content = content[:1_000_000]
            pii_results = pii_analyzer.analyze(text=truncated_content, language=app_config.pii_languages[0])
            return sorted(list({pii.entity_type for pii in pii_results})) if pii_results else None
        except Exception:
            return None

    def _extract_text_content(self) -> Optional[str]:
        """Extracts text content from various file types based on MIME type."""
        try:
            if not self.mime_type:
                return None

            # Avoid reading extremely large files into memory.
            if self.file_path.stat().st_size > MAX_CONTENT_EXTRACTION_SIZE_BYTES:
                return None

            # Do not attempt to extract text from binary formats like images/video/audio
            # as it will likely be meaningless and is computationally expensive.
            if self.mime_type.startswith(("image", "video", "audio")):
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
                # PyMuPDF can be noisy with errors from its underlying C library.
                # We'll suppress stderr to keep the console clean during processing.
                with pymupdf.suppress_stderr():
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

    def _get_pronom_id_with_fido(self) -> Optional[Tuple[str, str]]:
        """
        Uses Fido to get a more accurate file format identification (PRONOM ID).
        This is slower as it involves a subprocess call, so it's used as a fallback.
        Returns a tuple of (puid, mimetype) or None.
        """
        if not self.app_config.use_fido:
            return None
        try:
            # Fido writes its output to stdout. We capture it.
            # The '-q' flag makes the output cleaner (just the CSV).
            # The '-input' flag is more explicit for specifying the file path.
            result = subprocess.run(
                ["fido", "-q", "-input", str(self.file_path)],
                capture_output=True, text=True, check=True
            )
            # Fido output is a CSV: status,time,puid,formatname,signaturename,mimetype,basis,warning
            # We take the first line of output, as a file can have multiple matches.
            first_line = result.stdout.strip().splitlines()[0]
            parts = first_line.split(',')
            puid = parts[2].strip('"')
            mimetype = parts[5].strip('"')
            return puid, mimetype
        except FileNotFoundError:
            logging.error("The 'fido' command was not found. Please ensure 'opf-fido' is installed and in your system's PATH.")
            return None
        except subprocess.CalledProcessError as e:
            logging.warning(f"Fido failed to process {self.file_path}. It may be corrupt. Fido stderr: {e.stderr.strip()}")
            return None
        except IndexError:
            logging.warning(f"Fido returned unexpected output for {self.file_path}. Could not parse PRONOM ID.")
            return None

    def process(self) -> Generator[FileMetadata, None, None]:
        """
        Processes the file and returns a FileMetadata object.
        If the file is an archive, it yields metadata for each file within it.
        """
        try:
            # --- Early exit for excluded file types ---
            if self.file_path.name.lower() in DEFAULT_EXCLUDED_FILENAMES:
                return
            if self.file_path.suffix.lower() in DEFAULT_EXCLUDED_EXTENSIONS:
                return

            # Handle cases where the file might be deleted between scanning and processing
            if not self.file_path.exists():
                return

            if not self.file_path.is_file():
                return

            self.mime_type = magic.from_file(str(self.file_path), mime=True)
            pronom_id = None

            # If magic gives a generic result, try Fido for a more specific ID.
            if self.app_config.use_fido and self.mime_type in ("application/octet-stream", "inode/x-empty"):
                fido_result = self._get_pronom_id_with_fido()
                if fido_result:
                    pronom_id, fido_mimetype = fido_result
                    if fido_mimetype:
                        self.mime_type = fido_mimetype
            
            metadata_kwargs = {
                "path": self.virtual_path,
                "filename": self.file_path.name,
                "size_bytes": (stat_result := self.file_path.stat()).st_size,
                "mtime": stat_result.st_mtime,
                "crypto_hash": self._get_crypto_hash(),
                "mime_type": self.mime_type,
                "perceptual_hash": None,
                "content": None,
                "exif_data": None,
                "pronom_id": pronom_id,
                "content_embedding": None,
                "pii_types": None,
                "is_archived_file": self.is_archived,
            }

            # --- Handle Archives ---
            # For archives, we process the container file first, then its contents.
            if self.mime_type in ("application/zip", "application/x-tar", "application/gzip", "application/x-bzip2", "application/x-xz"):
                # First, yield the metadata for the archive file itself.
                yield FileMetadata(**metadata_kwargs)
                # Then, yield metadata for all files inside the archive.
                yield from self._process_archive()
                return # Stop here, as the archive's content has been handled.

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
                
            # For any text-based format, try to extract content.
            metadata_kwargs["content"] = self._extract_text_content()

            # If content was extracted, generate an embedding for it.
            if metadata_kwargs["content"]:
                embedding_model = get_embedding_model()
                embedding = embedding_model.encode(metadata_kwargs["content"])
                metadata_kwargs["content_embedding"] = embedding.tobytes()
            
            # --- PII Detection (Optimized) ---
            # For large text files, use the memory-efficient chunked scanner.
            # This avoids loading large content into memory just for PII analysis.
            if self.mime_type and self.mime_type.startswith("text/") and metadata_kwargs["size_bytes"] > 1_000_000:
                metadata_kwargs["pii_types"] = self._process_text_pii_in_chunks(self.file_path)
            # For smaller files where content was already extracted, analyze the content directly.
            elif metadata_kwargs["content"]:
                metadata_kwargs["pii_types"] = self._detect_pii(metadata_kwargs["content"])

            yield FileMetadata(**metadata_kwargs)
        except FileNotFoundError as e:
            # File was likely deleted between the scan and processing.
            logging.warning(f"File not found during processing (likely deleted): {self.file_path} - {e}")
            return
        except (IOError, PermissionError) as e:
            # Log other I/O related errors, like permission denied.
            logging.warning(f"I/O error processing file: {self.file_path} - {e}")
            return
        except Exception as e:
            # Catch any other unexpected errors during file processing
            logging.error(f"Unexpected error in FileProcessor for {self.file_path}: {e}", exc_info=True)
            return