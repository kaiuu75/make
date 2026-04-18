"""Config loader: YAML -> nested dataclasses with attribute access."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class _AttrDict(dict):
    """Nested dict with attribute-style access, kept serializable."""

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
        # Fallback only triggers when the attribute is not on the dataclass.
        return getattr(self.raw, key)


def load_config(path: str | Path) -> Config:
    p = Path(path)
    with p.open("r") as f:
        data = yaml.safe_load(f) or {}
    return Config(raw=_AttrDict(data), path=p.resolve())
