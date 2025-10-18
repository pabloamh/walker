 # walker/models.py
from typing import Optional

import attrs
from sqlalchemy import BLOB, Boolean, Column, Float, Integer, String, BigInteger, JSON
from sqlalchemy.orm import declarative_base

# --- Attrs class for data processing ---
# Using slots=True for memory and performance optimization, which is great
# when creating many objects.
@attrs.define(slots=True, frozen=True)
class FileMetadata:
    """A structured, immutable container for file metadata."""
    path: str
    filename: str
    size_bytes: int
    mtime: float
    crypto_hash: str
    mime_type: Optional[str]
    perceptual_hash: Optional[str]
    content: Optional[str]
    content_embedding: Optional[bytes]
    exif_data: Optional[str]
    pronom_id: Optional[str] = None
    pii_types: Optional[list[str]] = None
    is_archived_file: bool = False

# Forward-declare the Config class to avoid circular imports with worker.py
class Config:
    pass

# Default directories to exclude on Windows when scanning a root drive.
DEFAULT_WINDOWS_EXCLUDES = [
    "windows",
    "programdata",
    "program files",
    "program files (x86)",
    "pagefile.sys",
    "hiberfil.sys",
    "swapfile.sys",
    "$recycle.bin",
    "system volume information",
    "msocache",
]

# Default directories to exclude on macOS when scanning from the root.
DEFAULT_MACOS_EXCLUDES = [
    "/.DocumentRevisions-V100",
    "/.fseventsd",
    "/.Spotlight-V100",
    "/.Trashes",
    "/private",
    "/dev",
    "/System",
    "/Library",
    "/Applications",
    "/Users/*/Library",
]


# --- SQLAlchemy ORM class for database persistence ---
Base = declarative_base()

class FileIndex(Base):
    __tablename__ = 'file_index'

    id = Column(Integer, primary_key=True)
    path = Column(String, nullable=False, unique=True)
    filename = Column(String, nullable=False, index=True)
    size_bytes = Column(BigInteger, nullable=False, index=True)
    mtime = Column(Float, nullable=False) # Not frequently used in WHERE clauses, index is not critical
    crypto_hash = Column(String(64), nullable=False, index=True)  # For SHA-256
    mime_type = Column(String, nullable=True, index=True)
    perceptual_hash = Column(String, nullable=True, index=True)   # For images
    content = Column(String, nullable=True)                       # Full-text search is complex; not indexing by default
    pronom_id = Column(String, nullable=True, index=True)         # PRONOM ID from Fido
    exif_data = Column(JSON, nullable=True)                       # JSON columns are generally not indexed
    content_embedding = Column(BLOB, nullable=True)               # Vector indexes are special; not a standard index    
    pii_types = Column(JSON, nullable=True)                       # For storing a list of detected PII entity types
    is_archived_file = Column(Boolean, default=False, nullable=False) # Not used in queries, index is not needed

    @classmethod
    def from_metadata(cls, metadata: FileMetadata) -> "FileIndex":
        """Creates a FileIndex ORM instance from a FileMetadata object."""
        return cls(**attrs.asdict(metadata))

    def __repr__(self):
        return f"<FileIndex(filename='{self.filename}', path='{self.path}')>"