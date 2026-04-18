"""Binary prediction raster -> submission-ready GeoJSON.

This wraps the logic of the challenge's own ``submission_utils.raster_to_geojson``
with two additions:

* **Morphological cleanup** (opening + closing) of the binary mask, driven
  by the chosen thresholds.
* **Per-polygon ``time_step``** assignment when a YYMM raster is provided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
from shapely.geometry import shape

from ..inference.time_step import assign_time_steps_from_raster


def polygonize(
    prob: np.ndarray,
    *,
    transform,
    crs,
    threshold: float = 0.5,
    min_area_ha: float = 0.5,
    morph_open_px: int = 0,
    morph_close_px: int = 0,
    time_step_raster: np.ndarray | None = None,
) -> dict:
    """Turn a probability raster into a GeoJSON FeatureCollection in EPSG:4326."""
    binary = (prob >= threshold).astype(np.uint8)

    if morph_open_px and morph_open_px > 0:
        binary = binary_opening(binary.astype(bool), iterations=int(morph_open_px)).astype(np.uint8)
    if morph_close_px and morph_close_px > 0:
        binary = binary_closing(binary.astype(bool), iterations=int(morph_close_px)).astype(np.uint8)

    if binary.sum() == 0:
        return _empty_feature_collection()

    polygons: list[dict] = []
    for geom, value in shapes(binary, mask=binary, transform=transform):
        if value != 1:
            continue
        polygons.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {},
            }
        )

    if not polygons:
        return _empty_feature_collection()

    gdf = gpd.GeoDataFrame.from_features(polygons, crs=crs)

    # Area filter in UTM.
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    gdf = gdf[gdf_utm.area / 10_000 >= min_area_ha].reset_index(drop=True)

    if gdf.empty:
        return _empty_feature_collection()

    # Time-step assignment uses the *native grid* (same CRS as the raster).
    feats = json.loads(gdf.to_json())["features"]
    if time_step_raster is not None:
        # time_step_raster is on the *native* tile CRS/transform — pass the
        # features in native CRS for rasterize to work correctly. Rebuild.
        native = gdf.to_crs(crs)  # we read transform matching this CRS
        native_feats = json.loads(native.to_json())["features"]
        native_feats = assign_time_steps_from_raster(
            native_feats, time_step_raster, transform=transform
        )
        # Re-emit in EPSG:4326 with time_step attached.
        gdf_ts = gpd.GeoDataFrame.from_features(native_feats, crs=crs).to_crs("EPSG:4326")
        gdf_ts["time_step"] = [f["properties"].get("time_step") for f in native_feats]
        feats = json.loads(gdf_ts.to_json())["features"]
    else:
        gdf_out = gdf.to_crs("EPSG:4326")
        gdf_out["time_step"] = None
        feats = json.loads(gdf_out.to_json())["features"]

    return {"type": "FeatureCollection", "features": feats}


def merge_feature_collections(fcs: Iterable[dict]) -> dict:
    """Concatenate multiple FeatureCollections into one submission."""
    out: list[dict] = []
    for fc in fcs:
        for feat in fc.get("features", []):
            out.append(feat)
    return {"type": "FeatureCollection", "features": out}


def write_geojson(fc: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(fc, f)
    return p


def _empty_feature_collection() -> dict:
    return {"type": "FeatureCollection", "features": []}
