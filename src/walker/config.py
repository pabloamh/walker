# walker/config.py
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import attrs


@attrs.define(slots=True)
class Config:
    """Structured configuration for the walker application."""
    workers: int = 3
    db_batch_size: int = 500
    exclude_dirs: List[str] = attrs.field(factory=list)
    scan_dirs: List[str] = attrs.field(factory=list)
    embedding_model_path: Optional[str] = None


def load_config() -> Config:
    """
    Loads configuration from 'walker.toml'.
    It searches in the current working directory first, then in the script's directory.
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
        click.echo(f"Loading configuration from {config_path}")
        with config_path.open("rb") as f:
            config_data = tomllib.load(f)
    else:
        click.echo("No 'walker.toml' found. Using default settings.")

    # Get the [tool.walker] table from the TOML file
    walker_config = config_data.get("tool", {}).get("walker", {})

    return Config(
        workers=walker_config.get("workers", 3),
        db_batch_size=walker_config.get("db_batch_size", 500),
        exclude_dirs=walker_config.get("exclude_dirs", []),
        scan_dirs=walker_config.get("scan_dirs", []),
        embedding_model_path=walker_config.get("embedding_model_path"),
    )