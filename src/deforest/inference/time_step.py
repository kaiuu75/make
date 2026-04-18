"""Assign ``time_step`` (YYMM) to each predicted polygon.

Strategy
--------
For each polygon (already in EPSG:4326), rasterize it back onto the tile's
per-pixel YYMM raster and take the **mode** (most frequent non-zero value).
This keeps the assigned time_step consistent with the weak-label alerts
inside that polygon and avoids the median-across-sources rounding problems
at polygon boundaries.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from rasterio.features import geometry_mask
from shapely.geometry import shape


def assign_time_steps_from_raster(
    features: Iterable[dict],
    time_step_raster: np.ndarray,
    transform,
) -> list[dict]:
    """Add ``properties.time_step`` to each feature by polygon voting.

    Parameters
    ----------
    features
        List of GeoJSON-like dicts with at minimum a ``geometry`` key. Input
        geometries are expected in the SAME CRS as ``time_step_raster``.
    time_step_raster
        (H, W) int32 raster of YYMM values (0 = no alert).
    transform
        Affine transform of ``time_step_raster``.
    """
    h, w = time_step_raster.shape
    out: list[dict] = []
    for feat in features:
        geom = shape(feat["geometry"])
        mask = geometry_mask(
            [geom], transform=transform, invert=True, out_shape=(h, w)
        )
        values = time_step_raster[mask]
        nz = values[values > 0]
        ts: int | None = None
        if nz.size > 0:
            # Mode, break ties by most recent.
            vals, counts = np.unique(nz, return_counts=True)
            best = np.argmax(counts)
            ts = int(vals[best])
        feat = dict(feat)
        feat["properties"] = dict(feat.get("properties", {}) or {})
        feat["properties"]["time_step"] = ts
        out.append(feat)
    return out
