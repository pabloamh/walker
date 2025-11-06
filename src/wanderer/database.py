# wanderer/database.py
from sqlalchemy import create_engine
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from threading import Lock
from .models import Base
from . import config

engine = None
SessionLocal = None


# A lock to ensure only one thread writes to the DB at a time.
db_lock = Lock()

def init_db():
    """Creates the database tables."""
    global engine, SessionLocal
    if engine:
        return

    app_config = config.load_config()
    database_url = f"sqlite:///{app_config.database_path}"

    engine = create_engine(database_url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

def get_engine():
    return engine