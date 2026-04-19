"""Assign ``time_step`` (YYMM) to predicted polygons.

For each polygon (in the raster's native CRS) rasterize it onto the tile's
``(H, W) int32`` YYMM raster and take the mode of non-zero values as the
polygon's ``time_step``. This matches the polygon-level YYMM assignment used
by the zero-training baseline.
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
            vals, counts = np.unique(nz, return_counts=True)
            best = int(np.argmax(counts))
            ts = int(vals[best])
        feat = dict(feat)
        feat["properties"] = dict(feat.get("properties", {}) or {})
        feat["properties"]["time_step"] = ts
        out.append(feat)
    return out
