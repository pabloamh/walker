# tests/conftest.py
import pytest
from pathlib import Path
import shutil

@pytest.fixture(scope="function")
def test_env(tmp_path: Path, monkeypatch):
    """
    Creates a self-contained temporary environment for testing.

    This fixture provides a temporary directory structure that mimics the actual
    project, allowing tests to run in isolation without affecting the real
    database or configuration.

    Structure:
        tmp_path/
        ├── project_root/
        │   ├── src/
        │   │   └── walker/
        │   │       ├── walker.toml  (test config)
        │   │       └── models/
        │   └── test_data/          (files to be indexed)
        │       ├── file1.txt
        │       ├── file2.txt
        │       └── subdir/
        │           └── file3_dupe.txt
        └── home/
            └── .walker_db/         (location for the test database)
    """
    project_root = tmp_path / "project_root"
    src_walker_dir = project_root / "src" / "walker"
    test_data_dir = project_root / "test_data"
    subdir = test_data_dir / "subdir"

    # Create directories
    src_walker_dir.mkdir(parents=True)
    (src_walker_dir / "models").mkdir()
    subdir.mkdir(parents=True)

    # Create a test walker.toml
    (src_walker_dir / "walker.toml").write_text(f"""
[tool.walker]
scan_dirs = ["{test_data_dir.as_posix()}"]
workers = 1
pii_languages = ["en"]
    """)

    # Create some test files
    (test_data_dir / "file1.txt").write_text("This is the first file.")
    (test_data_dir / "file2.txt").write_text("This is the second file, it is unique.")
    (subdir / "file3_dupe.txt").write_text("This is the first file.") # Duplicate content

    # Monkeypatch the script's directory to make config/db loading work
    monkeypatch.setattr("walker.main.Path.home", lambda: tmp_path / "home")
    monkeypatch.setattr("walker.config.Path.home", lambda: tmp_path / "home")
    monkeypatch.setattr("walker.database.Path.home", lambda: tmp_path / "home")
    monkeypatch.setattr("walker.indexer.Path.home", lambda: tmp_path / "home")

    # Monkeypatch the location of the config file
    monkeypatch.setattr("walker.config.CONFIG_FILE_PATH", src_walker_dir / "walker.toml")

    yield project_root
    shutil.rmtree(project_root, ignore_errors=True)