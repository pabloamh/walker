# walker/config.py
import sys
from pathlib import Path
from typing import Any, Dict, List

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import attrs


@attrs.define(slots=True)
class Config:
    """Structured configuration for the walker application."""
    workers: int = 3
    exclude_dirs: List[str] = attrs.field(factory=list)


def load_config() -> Config:
    """
    Loads configuration from a 'walker.toml' file in the current directory.
    If the file doesn't exist, it returns a default configuration.
    """
    config_path = Path("walker.toml")
    config_data: Dict[str, Any] = {}

    if config_path.is_file():
        with config_path.open("rb") as f:
            config_data = tomllib.load(f)

    # Get the [tool.walker] table from the TOML file
    walker_config = config_data.get("tool", {}).get("walker", {})

    return Config(
        workers=walker_config.get("workers", 3),
        exclude_dirs=walker_config.get("exclude_dirs", []),
    )