"""Minimal YAML config loader with nested attribute access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class _AttrDict(dict):
    """Nested dict with ``cfg.key.subkey`` access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return _AttrDict(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


@dataclass
class Config:
    raw: _AttrDict
    path: Path

    def __getattr__(self, key: str) -> Any:
        return getattr(self.raw, key)


def load_config(path: str | Path) -> Config:
    p = Path(path).expanduser()
    with p.open("r") as f:
        data = yaml.safe_load(f) or {}
    return Config(raw=_AttrDict(data), path=p.resolve())


def merge_overrides(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into a copy of ``base`` and return it."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_overrides(out[key], value)
        else:
            out[key] = value
    return out
