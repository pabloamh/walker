# wanderer/database.py
from sqlalchemy import create_engine
from contextlib import contextmanager
import threading
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from .models import Base
from . import config

engine = None
SessionLocal = None
db_lock = threading.Lock()


def init_db(force_recreate: bool = False):
    """Creates the database tables."""
    global engine, SessionLocal
    if engine is not None and not force_recreate:
        return # Already initialized in this process

    # Load config within the function to ensure it's fresh for each process.
    app_config = config.load_config()
    database_url = f"sqlite:///{app_config.database_path}"

    engine = create_engine(database_url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    if not SessionLocal:
        init_db()
    db_session = SessionLocal() # type: ignore
    try:
        yield db_session
    finally:
        db_session.close()

def get_engine():
    return engine