# walker/config.py
import tomllib
import functools
import attrs
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_EXCLUDE_DIRS = [
    # --- General ---
    ".git",
    ".cache",
    "node_modules",
    "bower_components",
    "vendor",
    # --- Python ---
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "*.egg-info",
    # --- JS/Web Development ---
    ".idea",
    ".vscode",
    ".next",
    ".nuxt",
    # --- Other Build/Cache ---
    "target",
]


@attrs.define(slots=True)
class Config:
    """Structured configuration for the walker application."""
    workers: int = 3
    db_batch_size: int = 500
    exclude_dirs: List[str] = attrs.field(factory=lambda: list(DEFAULT_EXCLUDE_DIRS))
    scan_dirs: List[str] = attrs.field(factory=list)
    pii_languages: List[str] = attrs.field(factory=lambda: ["en"])
    memory_limit_gb: Optional[float] = None
    embedding_model_path: Optional[str] = None


@functools.lru_cache(maxsize=1)
def load_config() -> Config:
    """
    Loads configuration from 'walker.toml'.
    If not found, it returns a default configuration.
    """
    # Look for walker.toml in the current directory first, then in the script's directory.
    search_paths = [Path.cwd(), Path(__file__).parent]
    config_path = None
    for p in search_paths:
        if (p / "walker.toml").is_file():
            config_path = p / "walker.toml"
            break

    config_data: Dict[str, Any] = {}

    if config_path:
        with config_path.open("rb") as f:
            config_data = tomllib.load(f)

    # Get the [tool.walker] table from the TOML file
    walker_config = config_data.get("tool", {}).get("walker", {})

    return Config(
        workers=walker_config.get("workers", 3),
        db_batch_size=walker_config.get("db_batch_size", 500),
        exclude_dirs=list(set(DEFAULT_EXCLUDE_DIRS + walker_config.get("exclude_dirs", []))),
        scan_dirs=walker_config.get("scan_dirs", []),
        pii_languages=walker_config.get("pii_languages", ["en"]),
        memory_limit_gb=walker_config.get("memory_limit_gb"),
        embedding_model_path=walker_config.get("embedding_model_path"),
    )