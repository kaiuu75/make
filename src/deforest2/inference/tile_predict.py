"""Run the GBM tile-by-tile against the challenge dataset layout.

Entry point: :func:`predict_tile`. For a given tile it

1. picks the Sentinel-2 grid as the reference,
2. reprojects AEF + (optional) weak-label rasters onto it,
3. computes S1/S2 annual statistics (and optional multi-year drops),
4. packs features, runs the GBM, multiplies by the 2020 forest mask,
5. assembles a YYMM raster for ``time_step`` from the fused label dates,
6. returns a :class:`TilePrediction` with the probability raster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling

from ..data.align import reproject_multiband_to_grid, reproject_to_grid
from ..data.forest_mask import forest_mask_from_aef, forest_mask_from_ndvi
from ..data.paths import DataPaths
from ..data.readers import list_s2_months, read_aef
from ..features.aef import aef_features
from ..features.satellite import (
    pack_features,
    s1_annual_stats,
    s2_annual_stats,
    s2_worst_drop_across_years,
)
from ..features.terrain import terrain_features
from ..labels.fusion import fuse, FusedLabels
from ..labels.parsers import (
    WeakLabel,
    days_to_yymm_vectorized,
    parse_gladl,
    parse_glads2,
    parse_radd,
)
from ..models.gbm import PixelGBM


@dataclass
class TilePrediction:
    tile_id: str
    prob: np.ndarray          # (H, W) float32
    time_step: np.ndarray     # (H, W) int32, YYMM or 0
    crs: CRS
    transform: object
    fused: FusedLabels | None = None
    forest_mask: np.ndarray | None = None


def predict_tile(
    tile_id: str,
    paths: DataPaths,
    gbm: PixelGBM,
    *,
    split: str = "test",
    fusion_cfg: dict | None = None,
    feature_cfg: dict | None = None,
    year_base: int = 2020,
    year_last: int | None = None,
) -> TilePrediction:
    feature_cfg = feature_cfg or {}
    fusion_cfg = fusion_cfg or {}

    s2_dir = paths.s2_dir(tile_id, split=split)
    s2_entries = list_s2_months(s2_dir)
    if not s2_entries:
        raise FileNotFoundError(f"No Sentinel-2 data for tile {tile_id} under {s2_dir}")

    ref_year, ref_month = s2_entries[0]
    ref_tif = paths.s2_tif(tile_id, ref_year, ref_month, split=split)
    with rasterio.open(ref_tif) as src:
        ref_crs = src.crs
        ref_transform = src.transform
        ref_shape = (src.height, src.width)

    years_available = sorted({y for (y, _) in s2_entries})
    if year_base not in years_available:
        year_base = years_available[0]
    if year_last is None:
        year_last = years_available[-1]

    # AEF across every year that is present on disk (needed for multi-year drift).
    aef_by_year: dict[int, np.ndarray] = {}
    for y in years_available:
        p = paths.aef_tiff(tile_id, y, split=split)
        if not p.exists():
            continue
        data, profile = read_aef(p)
        aef_by_year[y] = reproject_multiband_to_grid(
            data,
            src_transform=profile["transform"], src_crs=profile["crs"],
            dst_transform=ref_transform, dst_crs=ref_crs,
            dst_shape=ref_shape, resampling=Resampling.bilinear,
        )
    if not aef_by_year:
        raise FileNotFoundError(f"No AEF years available for tile {tile_id}")

    aef_feats = aef_features(
        aef_by_year, multi_year_drift=bool(feature_cfg.get("aef_multi_year_drift", True))
    )

    grid = dict(ref_transform=ref_transform, ref_crs=ref_crs, ref_shape=ref_shape)
    pct = feature_cfg.get("s2_percentiles", [10, 90])
    percentiles = tuple(pct) if pct else None
    include_std = bool(feature_cfg.get("s2_intra_year_std", True))

    s1_base = s1_annual_stats(paths.s1_dir(tile_id, split=split), year_base, **grid)
    s1_last = s1_annual_stats(paths.s1_dir(tile_id, split=split), year_last, **grid)
    s2_base = s2_annual_stats(
        s2_dir, year_base,
        percentiles=percentiles, include_intra_year_std=include_std, **grid,
    )
    s2_last = s2_annual_stats(
        s2_dir, year_last,
        percentiles=percentiles, include_intra_year_std=include_std, **grid,
    )

    s2_multi = None
    if feature_cfg.get("s2_worst_drop", True):
        s2_multi = s2_worst_drop_across_years(
            s2_dir,
            year_base=year_base,
            years=years_available,
            ref_transform=ref_transform,
            ref_crs=ref_crs,
            ref_shape=ref_shape,
        )

    terr = None
    if feature_cfg.get("use_terrain", False):
        terr = terrain_features(
            ref_transform, ref_crs, ref_shape,
            cache_dir=str(feature_cfg.get("terrain_cache_dir", "cache/dem")),
        )

    X, _names = pack_features(
        aef_feats, s2_base, s2_last, s1_base, s1_last,
        s2_multi=s2_multi, terrain=terr,
    )

    prob_flat = gbm.predict_proba(X)
    prob = prob_flat.reshape(ref_shape).astype(np.float32)

    forest_mask = _build_forest_mask_2020(aef_by_year, year_base, s2_base)
    prob *= forest_mask.astype(np.float32)

    fused: FusedLabels | None = None
    time_step = np.zeros(ref_shape, dtype=np.int32)
    if split == "train":
        fused = _load_and_fuse_labels(
            tile_id, paths,
            ref_crs=ref_crs, ref_transform=ref_transform, ref_shape=ref_shape,
            fusion_cfg=fusion_cfg,
        )
        if fused is not None:
            time_step = days_to_yymm_vectorized(fused.median_days).astype(np.int32)

    return TilePrediction(
        tile_id=tile_id,
        prob=prob,
        time_step=time_step,
        crs=ref_crs,
        transform=ref_transform,
        fused=fused,
        forest_mask=forest_mask.astype(bool),
    )


# ---------------------------------------------------------------------------
# Helpers exposed for the training script
# ---------------------------------------------------------------------------


def _build_forest_mask_2020(
    aef_by_year: dict[int, np.ndarray],
    year_base: int,
    s2_base: dict[str, np.ndarray] | None,
) -> np.ndarray:
    ndvi_med = None if s2_base is None else s2_base.get("median_ndvi")
    if year_base in aef_by_year:
        try:
            return forest_mask_from_aef(aef_by_year[year_base], ndvi_2020_median=ndvi_med)
        except Exception:  # pragma: no cover - defensive
            pass
    if ndvi_med is not None:
        return forest_mask_from_ndvi(ndvi_med)
    any_year = next(iter(aef_by_year.values()))
    return np.ones(any_year.shape[1:], dtype=bool)


def _load_and_fuse_labels(
    tile_id: str,
    paths: DataPaths,
    *,
    ref_crs,
    ref_transform,
    ref_shape: tuple[int, int],
    fusion_cfg: dict,
) -> FusedLabels | None:
    sources: dict[str, WeakLabel] = {}

    radd_path = paths.radd(tile_id)
    if radd_path.exists():
        with rasterio.open(radd_path) as src:
            radd_raw = src.read(1)
            radd_aligned = reproject_to_grid(
                radd_raw,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.int32,
            )
        sources["radd"] = parse_radd(radd_aligned)

    gladl_combined: WeakLabel | None = None
    for yy in range(20, 30):
        alert_p = paths.gladl_alert(tile_id, yy)
        date_p = paths.gladl_date(tile_id, yy)
        if not (alert_p.exists() and date_p.exists()):
            continue
        with rasterio.open(alert_p) as src_a, rasterio.open(date_p) as src_d:
            alert_aligned = reproject_to_grid(
                src_a.read(1), src_transform=src_a.transform, src_crs=src_a.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.uint8,
            )
            date_aligned = reproject_to_grid(
                src_d.read(1), src_transform=src_d.transform, src_crs=src_d.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.uint16,
            )
        wl = parse_gladl(alert_aligned, date_aligned, yy=yy)
        if gladl_combined is None:
            gladl_combined = wl
        else:
            mask = wl.confidence > gladl_combined.confidence
            gladl_combined.confidence[mask] = wl.confidence[mask]
            gladl_combined.days[mask] = wl.days[mask]
    if gladl_combined is not None:
        sources["gladl"] = gladl_combined

    a_p = paths.glads2_alert(tile_id)
    d_p = paths.glads2_date(tile_id)
    if a_p.exists() and d_p.exists():
        with rasterio.open(a_p) as src_a, rasterio.open(d_p) as src_d:
            alert_aligned = reproject_to_grid(
                src_a.read(1), src_transform=src_a.transform, src_crs=src_a.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.uint8,
            )
            date_aligned = reproject_to_grid(
                src_d.read(1), src_transform=src_d.transform, src_crs=src_d.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.uint16,
            )
        sources["glads2"] = parse_glads2(alert_aligned, date_aligned)

    if not sources:
        return None

    return fuse(
        sources,
        agreement_threshold=fusion_cfg.get("agreement_threshold", 0.7),
        single_threshold=fusion_cfg.get("single_threshold", 0.9),
    )
