"""Configuration loader and validation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If parsing fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def merge_configs(base_cfg: Dict[str, Any], override_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override configuration into base configuration.

    Args:
        base_cfg: Default configuration dictionary.
        override_cfg: User or experiment override dictionary.

    Returns:
        Merged configuration dictionary.
    """
    merged = base_cfg.copy()
    for key, val in override_cfg.items():
        if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], val)
        else:
            merged[key] = val
    return merged


def get_default_configs_dir() -> Path:
    """Return the absolute path to the configs directory."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    return project_root / "configs"
