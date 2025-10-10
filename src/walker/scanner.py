# walker/scanner.py
import fnmatch
import os
import sys
from pathlib import Path
from typing import Generator, Union, Set

def scan_directory(root_path: Union[str, Path], exclude: Set[str]) -> Generator[Path, None, None]:
    """
    Recursively yields all file paths in a directory as Path objects.
    Handles permission errors and supports glob patterns for exclusions.
    """
    p = Path(root_path)
    
    # Separate glob patterns from direct name matches for efficiency
    glob_patterns = {pattern for pattern in exclude if "*" in pattern or "?" in pattern}
    direct_excludes = exclude - glob_patterns

    try:
        for entry in p.iterdir():
            try:
                # Prepare both the name and the full path for checking against exclusions.
                entry_name_lower = entry.name.lower()
                # Use abspath instead of resolve() to avoid resolving symlinks, which could
                # lead to incorrect path validation against the root scan directories.
                entry_path_lower = os.path.normcase(os.path.abspath(str(entry)))

                # Check for exclusion
                if (entry_name_lower in direct_excludes or
                    entry_path_lower in direct_excludes or
                    any(fnmatch.fnmatch(entry_name_lower, pattern) for pattern in glob_patterns) or
                    any(fnmatch.fnmatch(entry_path_lower, pattern) for pattern in glob_patterns)):
                    continue

                if entry.is_dir(follow_symlinks=False):
                    yield from scan_directory(entry, exclude)
                elif entry.is_file(follow_symlinks=False):
                    yield entry
            except FileNotFoundError:
                # This can happen if a file is deleted while scanning.
                continue
    except PermissionError:
        # Silently skip directories that cannot be read.
        # You could add a logging statement here if desired.
        pass
    except Exception as e:
        # Catch other potential filesystem errors
        print(f"Error scanning {p}: {e}", file=sys.stderr)