# walker/scanner.py
from pathlib import Path
from typing import Generator, Union

def scan_directory(root_path: Union[str, Path]) -> Generator[Path, None, None]:
    """Recursively yields all file paths in a directory as Path objects."""
    p = Path(root_path)
    for entry in p.iterdir():
        if entry.is_dir(follow_symlinks=False):
            yield from scan_directory(entry)
        elif entry.is_file(follow_symlinks=False):
            yield entry