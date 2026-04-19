"""Dataset path helpers for the osapiens Makeathon 2026 layout.

Mirrors the layout documented in `makeathon26/challenge.ipynb`::

    sentinel-1/<split>/{tile_id}__s1_rtc/{tile_id}__s1_rtc_{YYYY}_{M}_{orbit}.tif
    sentinel-2/<split>/{tile_id}__s2_l2a/{tile_id}__s2_l2a_{YYYY}_{M}.tif
    aef-embeddings/<split>/{tile_id}_{YYYY}.tiff
    labels/train/
        radd/radd_{tile_id}_labels.tif
        gladl/gladl_{tile_id}_alert{YY}.tif
        gladl/gladl_{tile_id}_alertDate{YY}.tif
        glads2/glads2_{tile_id}_alert.tif
        glads2/glads2_{tile_id}_alertDate.tif
    metadata/{train,test}_tiles.geojson
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DataPaths:
    """Resolve challenge-dataset file paths from a single root directory."""

    root: Path
    s1_subdir: str = "sentinel-1"
    s2_subdir: str = "sentinel-2"
    aef_subdir: str = "aef-embeddings"
    labels_subdir: str = "labels/train"

    # --- per-tile directories / files --------------------------------------

    def s1_dir(self, tile_id: str, *, split: str = "train") -> Path:
        return Path(self.root) / self.s1_subdir / split / f"{tile_id}__s1_rtc"

    def s2_dir(self, tile_id: str, *, split: str = "train") -> Path:
        return Path(self.root) / self.s2_subdir / split / f"{tile_id}__s2_l2a"

    def s2_tif(self, tile_id: str, year: int, month: int, *, split: str = "train") -> Path:
        return self.s2_dir(tile_id, split=split) / f"{tile_id}__s2_l2a_{year}_{month}.tif"

    def s1_tif(
        self,
        tile_id: str,
        year: int,
        month: int,
        orbit: str,
        *,
        split: str = "train",
    ) -> Path:
        return self.s1_dir(tile_id, split=split) / f"{tile_id}__s1_rtc_{year}_{month}_{orbit}.tif"

    def aef_tiff(self, tile_id: str, year: int, *, split: str = "train") -> Path:
        return Path(self.root) / self.aef_subdir / split / f"{tile_id}_{year}.tiff"

    # --- labels (train split only) ----------------------------------------

    def radd(self, tile_id: str) -> Path:
        return Path(self.root) / self.labels_subdir / "radd" / f"radd_{tile_id}_labels.tif"

    def gladl_alert(self, tile_id: str, yy: int) -> Path:
        return Path(self.root) / self.labels_subdir / "gladl" / f"gladl_{tile_id}_alert{yy:02d}.tif"

    def gladl_date(self, tile_id: str, yy: int) -> Path:
        return (
            Path(self.root)
            / self.labels_subdir
            / "gladl"
            / f"gladl_{tile_id}_alertDate{yy:02d}.tif"
        )

    def glads2_alert(self, tile_id: str) -> Path:
        return Path(self.root) / self.labels_subdir / "glads2" / f"glads2_{tile_id}_alert.tif"

    def glads2_date(self, tile_id: str) -> Path:
        return Path(self.root) / self.labels_subdir / "glads2" / f"glads2_{tile_id}_alertDate.tif"


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------


def list_tiles(metadata_geojson: str | Path) -> list[str]:
    """Read a ``{train,test}_tiles.geojson`` and return tile ids."""
    p = Path(metadata_geojson)
    if not p.exists():
        return []
    with p.open("r") as f:
        fc = json.load(f)
    out: list[str] = []
    for feat in fc.get("features", []) or []:
        props = feat.get("properties", {}) or {}
        name = props.get("name") or props.get("tile_id") or props.get("id")
        if name:
            out.append(str(name))
    return out


def discover_tiles(root: Path, *, split: str = "train") -> list[str]:
    """Fallback tile discovery: scan the Sentinel-2 directory of ``split``."""
    s2_split = Path(root) / "sentinel-2" / split
    if not s2_split.exists():
        return []
    ids: set[str] = set()
    for sub in s2_split.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        if name.endswith("__s2_l2a"):
            ids.add(name[: -len("__s2_l2a")])
    return sorted(ids)


def write_tiles_geojson(tile_ids: Iterable[str], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    feats = [
        {"type": "Feature", "properties": {"name": tid}, "geometry": None}
        for tid in tile_ids
    ]
    with p.open("w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return p


def region_of(tile_id: str, *, prefix_chars: int = 5) -> str:
    """Return the MGRS-prefix region id used for cross-region validation.

    The first ``prefix_chars`` characters of ``tile_id`` encode the MGRS grid
    cell (e.g. ``18NWG`` for tile ``18NWG_6_6``). Tiles sharing that prefix
    are spatially adjacent and tend to come from the same biome.
    """
    return tile_id[:prefix_chars]
