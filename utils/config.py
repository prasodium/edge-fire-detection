"""Typed configuration loading for the Edge Fire Detection System.

All runtime tuning lives in configs/*.yaml. This module loads, validates and
caches them so the rest of the codebase never touches YAML directly.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class ConfigError(RuntimeError):
    """Raised when a config file is missing or malformed."""


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a top-level mapping")
    return data


@dataclass(frozen=True)
class AppConfig:
    """Aggregated, read-only view over every configs/*.yaml file."""

    system: dict[str, Any] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    alarm: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)

    def active_model_spec(self) -> dict[str, Any]:
        key = self.model["active_model"]
        try:
            return self.model["models"][key]
        except KeyError as exc:
            raise ConfigError(f"active_model '{key}' not present in models: section") from exc

    def project_root(self) -> Path:
        return CONFIG_DIR.parent


@functools.lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load and cache all configuration files for the process lifetime.

    Cached because YAML parsing happens on the hot path of module import in
    several places (camera, inference, dashboard) and the files do not
    change at runtime - restart the service to pick up edits.
    """
    system_raw = _load_yaml("system.yaml")
    return AppConfig(
        system=system_raw["system"] | {"paths": system_raw["paths"]},
        camera=_load_yaml("camera.yaml")["camera"],
        model=_load_yaml("model.yaml"),
        decision=_load_yaml("decision.yaml"),
        alarm=_load_yaml("alarm.yaml"),
        dataset=_load_yaml("dataset.yaml"),
    )


def reload_config() -> AppConfig:
    """Force a fresh read of all config files, bypassing the cache."""
    load_config.cache_clear()
    return load_config()
