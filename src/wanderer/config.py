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
    """Structured configuration for the wanderer application."""
    workers: int = 3
    db_batch_size: int = 500
    exclude_dirs: List[str] = attrs.field(factory=lambda: list(DEFAULT_EXCLUDE_DIRS))
    scan_dirs: List[str] = attrs.field(factory=list)
    pii_languages: List[str] = attrs.field(factory=lambda: ["en"])
    memory_limit_gb: Optional[float] = None
    embedding_model_path: Optional[str] = "models/all-MiniLM-L6-v2"
    use_fido: bool = False
    extract_text_on_scan: bool = True
    compute_perceptual_hash: bool = True
    archive_exclude_extensions: List[str] = attrs.field(factory=lambda: [".epub", ".cbz", ".cbr"])


@functools.lru_cache(maxsize=1)
def load_config_with_path() -> tuple[Config, Optional[Path]]:
    """
    Loads configuration from 'wanderer.toml'.
    If not found, it returns a default configuration and None for the path.
    Returns a tuple of (Config, Optional[Path]).
    """
    # Look for wanderer.toml in the current directory first. This is the most
    # intuitive location for a user-facing configuration file.
    search_paths = [Path.cwd(), Path(__file__).parent]
    config_path = None
    for p in search_paths:
        if (p / "wanderer.toml").is_file():
            config_path = p / "wanderer.toml"
            break

    config_data: Dict[str, Any] = {}

    if config_path:
        with config_path.open("rb") as f:
            config_data = tomllib.load(f)

    # Get the [tool.wanderer] table from the TOML file
    wanderer_config = config_data.get("tool", {}).get("wanderer", {})
    
    # Default extensions for archives that should not be extracted
    default_archive_excludes = [".epub", ".cbz", ".cbr"]

    loaded_config = Config(
        workers=wanderer_config.get("workers", 3),
        db_batch_size=wanderer_config.get("db_batch_size", 500),
        exclude_dirs=list(set(DEFAULT_EXCLUDE_DIRS + wanderer_config.get("exclude_dirs", []))),
        scan_dirs=wanderer_config.get("scan_dirs", []),
        pii_languages=wanderer_config.get("pii_languages", ["en"]),
        memory_limit_gb=wanderer_config.get("memory_limit_gb"),
        embedding_model_path=wanderer_config.get("embedding_model_path", "models/all-MiniLM-L6-v2"),
        use_fido=wanderer_config.get("use_fido", False),
        extract_text_on_scan=wanderer_config.get("extract_text_on_scan", True),
        compute_perceptual_hash=wanderer_config.get("compute_perceptual_hash", True),
        archive_exclude_extensions=list(set(default_archive_excludes + wanderer_config.get("archive_exclude_extensions", []))),
    )
    return loaded_config, config_path

def load_config() -> Config:
    """
    Loads configuration from 'wanderer.toml'.
    This is a convenience wrapper around load_config_with_path.
    """
    return load_config_with_path()[0]