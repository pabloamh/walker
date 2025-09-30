# walker/scanner.py
from pathlib import Path
from typing import Generator, Union, Set

def scan_directory(root_path: Union[str, Path], exclude: Set[str]) -> Generator[Path, None, None]:
    """Recursively yields all file paths in a directory as Path objects."""
    p = Path(root_path)
    for entry in p.iterdir():
        if entry.is_dir(follow_symlinks=False):
            # Skip directory if its name is in the exclusion set (case-insensitive)
            if entry.name.lower() in exclude:
                continue
            yield from scan_directory(entry, exclude)
        elif entry.is_file(follow_symlinks=False):
            yield entry