# walker/database.py
from sqlalchemy import create_engine
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from threading import Lock
from .models import Base

DATABASE_URL = "sqlite:///file_indexer.db"

# The check_same_thread=False is needed for SQLite when used with multiple threads.
# The connect_args dictionary is passed to the DB-API's connect() function.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# A thread-safe way to get a session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# A lock to ensure only one thread writes to the DB at a time.
db_lock = Lock()

def init_db():
    """Creates the database tables."""
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()