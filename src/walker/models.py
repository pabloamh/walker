# walker/models.py
from typing import Optional

import attrs
from sqlalchemy import BLOB, Column, Integer, String, BigInteger
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
    exif_data: Optional[str]
    content_embedding: Optional[bytes]


# --- SQLAlchemy ORM class for database persistence ---
Base = declarative_base()

class FileIndex(Base):
    __tablename__ = 'file_index'

    id = Column(Integer, primary_key=True)
    path = Column(String, nullable=False, unique=True)
    filename = Column(String, nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    mtime = Column(BigInteger, nullable=False)
    crypto_hash = Column(String(64), nullable=False)  # For SHA-256
    mime_type = Column(String, nullable=True)
    perceptual_hash = Column(String, nullable=True)   # For images
    content = Column(String, nullable=True)           # For documents
    exif_data = Column(String, nullable=True)         # For image EXIF data
    content_embedding = Column(BLOB, nullable=True)   # For text embeddings

    @classmethod
    def from_metadata(cls, metadata: FileMetadata) -> "FileIndex":
        """Creates a FileIndex ORM instance from a FileMetadata object."""
        return cls(**attrs.asdict(metadata))

    def __repr__(self):
        return f"<FileIndex(filename='{self.filename}', path='{self.path}')>"