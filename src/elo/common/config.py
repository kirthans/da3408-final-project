"""Load pipeline parameters and explicitly requested environment settings."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    """Read a YAML parameter mapping.

    Args:
        path: Parameter file, defaulting to the repository's params.yaml.

    Returns:
        The decoded top-level mapping.

    Raises:
        ValueError: The YAML root is not a mapping."""
    with Path(path).open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    if not isinstance(params, dict):
        raise ValueError("Parameters must be a YAML mapping")
    return params


def get_env(name: str, default: str | None = None) -> str:
    """Read an environment setting with an optional fallback.

    Args:
        name: Environment variable name.
        default: Fallback value; None makes the setting required.

    Returns:
        The configured or fallback string.

    Raises:
        ValueError: A required setting is absent."""
    value = os.environ.get(name, default)
    if value is None:
        raise ValueError(f"Required environment variable is missing: {name}")
    return value
