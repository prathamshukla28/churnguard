"""Configuration loading.

Centralises every tunable knob in ``config/config.yaml`` so that the data,
training and serving code never hard-code paths or hyper-parameters. Paths in
the config are resolved relative to the project root, which makes the project
runnable from any working directory (CLI, tests, Docker, CI).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

# Project root = the directory that contains this package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Typed view over the YAML config with path resolution helpers."""

    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup, e.g. ``cfg.get("model", "cv_folds")``."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, *keys: str) -> Path:
        """Resolve a config value that holds a project-relative path."""
        value = self.get(*keys)
        if value is None:
            raise KeyError(f"No path configured at {'.'.join(keys)}")
        return (PROJECT_ROOT / str(value)).resolve()


@cache
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the project configuration."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)
