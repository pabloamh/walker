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
                # Prepare both the name and the full path for checking against exclusions,
                # ensuring they are normalized for case-insensitive comparison.
                entry_name_norm = os.path.normcase(entry.name)
                # Use abspath instead of resolve() to avoid resolving symlinks, which could
                # lead to incorrect path validation against the root scan directories.
                entry_path_norm = os.path.normcase(os.path.abspath(str(entry)))
                
                # --- Exclusion Check ---
                # 1. Check if the simple name is in the direct exclusion list (e.g., "node_modules").
                # 2. Check if the full path is in the direct exclusion list (e.g., "/media/alpha/programdata").
                # 3. Check if the name or full path matches any glob patterns.
                is_excluded = (
                    entry_name_norm in direct_excludes or
                    entry_path_norm in direct_excludes or
                    any(fnmatch.fnmatch(entry_name_norm, p) for p in glob_patterns) or
                    any(fnmatch.fnmatch(entry_path_norm, p) for p in glob_patterns)
                )
                if is_excluded:
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