"""Sentinel-1 and Sentinel-2 annual statistics with richer per-year features.

Added: in addition to ``make/``'s median / min / slope, we now
compute

* S2: per-year p10/p90 NDVI and intra-year NDVI std (toggled by config).
* S2: "worst-drop" aggregates across **all** available years — the per-pixel
  minimum NDVI ever observed after ``year_base`` and the year in which it
  occurred (``s2_worst_drop`` / ``s2_year_of_drop``).
* S1: mean / std / min of VV in dB for the chosen year (unchanged).

All features are packed into ``(H*W, F)`` in :func:`pack_features`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from ..data.align import reproject_multiband_to_grid, reproject_to_grid
from ..data.readers import read_s1, read_s2, list_s1_months, list_s2_months


# ---- NDVI / NDMI helpers ---------------------------------------------------


def ndvi_from_s2(bands: np.ndarray) -> np.ndarray:
    """NDVI = (NIR − RED) / (NIR + RED) using B08 (index 7) and B04 (index 3)."""
    nir = bands[7].astype(np.float32)
    red = bands[3].astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / np.maximum(denom, 1e-6), 0.0)
    return ndvi.astype(np.float32)


def ndmi_from_s2(bands: np.ndarray) -> np.ndarray:
    """NDMI = (NIR − SWIR) / (NIR + SWIR) using B08 and B11 (index 10)."""
    nir = bands[7].astype(np.float32)
    swir = bands[10].astype(np.float32)
    denom = nir + swir
    ndmi = np.where(denom > 0, (nir - swir) / np.maximum(denom, 1e-6), 0.0)
    return ndmi.astype(np.float32)


# ---- Annual statistics -----------------------------------------------------


def s2_annual_stats(
    s2_dir: str | Path,
    year: int,
    *,
    ref_transform=None,
    ref_crs=None,
    ref_shape: tuple[int, int] | None = None,
    percentiles: tuple[int, int] | None = (10, 90),
    include_intra_year_std: bool = True,
) -> dict[str, np.ndarray] | None:
    """Compute NDVI / NDMI statistics for ``year`` (median + min + slope + extras)."""
    months = [m for (y, m) in list_s2_months(s2_dir) if y == year]
    if not months:
        return None

    ndvi_stack, ndmi_stack = [], []
    for m in months:
        tif = Path(s2_dir) / f"{Path(s2_dir).name}_{year}_{m}.tif"
        if not tif.exists():
            continue
        bands, prof = read_s2(tif)
        if ref_transform is None or ref_crs is None or ref_shape is None:
            ref_transform = prof["transform"]
            ref_crs = prof["crs"]
            ref_shape = prof["shape"]
        elif bands.shape[1:] != ref_shape or prof["transform"] != ref_transform:
            bands = reproject_multiband_to_grid(
                bands,
                src_transform=prof["transform"], src_crs=prof["crs"],
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.bilinear,
            )
        ndvi_stack.append(ndvi_from_s2(bands))
        ndmi_stack.append(ndmi_from_s2(bands))

    if not ndvi_stack:
        return None
    ndvi = np.stack(ndvi_stack, axis=0)
    ndmi = np.stack(ndmi_stack, axis=0)

    out: dict[str, np.ndarray] = {
        "median_ndvi": np.nanmedian(ndvi, axis=0).astype(np.float32),
        "median_ndmi": np.nanmedian(ndmi, axis=0).astype(np.float32),
        "min_ndvi": np.nanmin(ndvi, axis=0).astype(np.float32),
    }

    # NDVI slope over the month index (least squares).
    t = np.arange(ndvi.shape[0], dtype=np.float32)
    if ndvi.shape[0] >= 2:
        t_mean = t.mean()
        v_mean = np.nanmean(ndvi, axis=0)
        num = np.nansum((t - t_mean)[:, None, None] * (ndvi - v_mean[None]), axis=0)
        den = np.nansum((t - t_mean) ** 2)
        out["slope_ndvi"] = (num / max(den, 1e-6)).astype(np.float32)
    else:
        out["slope_ndvi"] = np.zeros_like(out["median_ndvi"])

    if percentiles and ndvi.shape[0] >= 2:
        lo, hi = percentiles
        out[f"p{lo}_ndvi"] = np.nanpercentile(ndvi, lo, axis=0).astype(np.float32)
        out[f"p{hi}_ndvi"] = np.nanpercentile(ndvi, hi, axis=0).astype(np.float32)

    if include_intra_year_std and ndvi.shape[0] >= 2:
        out["std_ndvi"] = np.nanstd(ndvi, axis=0).astype(np.float32)

    return out


def s2_worst_drop_across_years(
    s2_dir: str | Path,
    *,
    year_base: int,
    years: list[int],
    ref_transform,
    ref_crs,
    ref_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Per-pixel worst NDVI drop vs ``year_base`` across all other ``years``.

    Returns an empty dict when only the base year is available.
    """
    base_stats = s2_annual_stats(
        s2_dir, year_base,
        ref_transform=ref_transform, ref_crs=ref_crs, ref_shape=ref_shape,
        percentiles=None, include_intra_year_std=False,
    )
    if base_stats is None:
        return {}
    base_med = base_stats["median_ndvi"]

    later = [y for y in years if y != year_base]
    min_ndvi_later: list[np.ndarray] = []
    year_list: list[int] = []
    for y in later:
        stats = s2_annual_stats(
            s2_dir, y,
            ref_transform=ref_transform, ref_crs=ref_crs, ref_shape=ref_shape,
            percentiles=None, include_intra_year_std=False,
        )
        if stats is None:
            continue
        min_ndvi_later.append(stats["min_ndvi"])
        year_list.append(y)

    if not min_ndvi_later:
        return {}

    stacked = np.stack(min_ndvi_later, axis=0)        # (Y, H, W)
    drops = stacked - base_med[None]                  # negative = drop
    worst = drops.min(axis=0).astype(np.float32)      # most negative value
    idx = drops.argmin(axis=0)
    year_of = np.array(year_list, dtype=np.int16)[idx].astype(np.int16)
    return {"s2_worst_drop": worst, "s2_year_of_drop": year_of}


def s1_annual_stats(
    s1_dir: str | Path,
    year: int,
    *,
    ref_transform=None,
    ref_crs=None,
    ref_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray] | None:
    """Mean / std / min of VV (dB) for ``year`` across all orbits."""
    entries = [(y, m, o) for (y, m, o) in list_s1_months(s1_dir) if y == year]
    if not entries:
        return None

    stack = []
    for (y, m, o) in entries:
        tif = Path(s1_dir) / f"{Path(s1_dir).name}_{y}_{m}_{o}.tif"
        if not tif.exists():
            continue
        vv, prof = read_s1(tif)
        if ref_transform is not None and ref_crs is not None and ref_shape is not None:
            if vv.shape != ref_shape or prof["transform"] != ref_transform:
                vv = reproject_to_grid(
                    vv,
                    src_transform=prof["transform"], src_crs=prof["crs"],
                    dst_transform=ref_transform, dst_crs=ref_crs,
                    dst_shape=ref_shape, resampling=Resampling.bilinear,
                    src_nodata=np.nan, dst_nodata=np.nan, dtype=np.float32,
                )
        stack.append(vv)
    if not stack:
        return None
    if not all(a.shape == stack[0].shape for a in stack):
        target = stack[0].shape
        stack = [a if a.shape == target else _crop_or_pad(a, target) for a in stack]
    arr = np.stack(stack, axis=0)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "mean_vv": np.nanmean(arr, axis=0).astype(np.float32),
            "std_vv": np.nanstd(arr, axis=0).astype(np.float32),
            "min_vv": np.nanmin(arr, axis=0).astype(np.float32),
        }


def _crop_or_pad(arr: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Centre-crop or zero-pad a 2-D array to ``target`` shape."""
    h, w = arr.shape
    th, tw = target
    fill = np.nan if np.issubdtype(arr.dtype, np.floating) else 0
    out = np.full(target, fill, dtype=arr.dtype)
    sh = min(h, th); sw = min(w, tw)
    out[:sh, :sw] = arr[:sh, :sw]
    return out


# ---- Feature packing -------------------------------------------------------


def pack_features(
    aef_feats: dict[str, np.ndarray],
    s2_base: dict[str, np.ndarray] | None,
    s2_last: dict[str, np.ndarray] | None,
    s1_base: dict[str, np.ndarray] | None,
    s1_last: dict[str, np.ndarray] | None,
    *,
    s2_multi: dict[str, np.ndarray] | None = None,
    terrain: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Stack all features into ``(H*W, F)``. Returns ``(X, feature_names)``."""
    layers: list[np.ndarray] = []
    names: list[str] = []

    base = aef_feats["aef_base"]
    last = aef_feats["aef_last"]
    delta = aef_feats["aef_delta"]
    h, w = base.shape[1:]
    n = h * w

    def _append_2d(arr: np.ndarray, name: str) -> None:
        layers.append(arr.reshape(-1, 1))
        names.append(name)

    def _append_multi(arr: np.ndarray, prefix: str) -> None:
        c = arr.shape[0]
        layers.append(arr.reshape(c, -1).T)
        names.extend(f"{prefix}_{i}" for i in range(c))

    _append_multi(base, "aef_base")
    _append_multi(last, "aef_last")
    _append_multi(delta, "aef_delta")
    _append_2d(aef_feats["aef_norm"], "aef_norm")
    _append_2d(aef_feats["aef_cos_dist"], "aef_cos_dist")
    if "aef_max_drift" in aef_feats:
        _append_2d(aef_feats["aef_max_drift"], "aef_max_drift")
        _append_2d(aef_feats["aef_year_of_drift"].astype(np.float32), "aef_year_of_drift")

    def add_scalar(m: dict[str, np.ndarray] | None, keys: list[str], prefix: str) -> None:
        for k in keys:
            v = None if m is None else m.get(k)
            if v is None:
                v = np.zeros((h, w), dtype=np.float32)
            _append_2d(v, f"{prefix}_{k}")

    add_scalar(s1_base, ["mean_vv", "std_vv", "min_vv"], "s1_base")
    add_scalar(s1_last, ["mean_vv", "std_vv", "min_vv"], "s1_last")

    s2_keys = ["median_ndvi", "median_ndmi", "min_ndvi", "slope_ndvi"]
    if s2_base is not None:
        for extra in ["p10_ndvi", "p90_ndvi", "std_ndvi"]:
            if extra in s2_base and extra not in s2_keys:
                s2_keys.append(extra)
    add_scalar(s2_base, s2_keys, "s2_base")
    add_scalar(s2_last, s2_keys, "s2_last")

    # Deltas
    for stats_base_d, stats_last_d, keys, prefix in [
        (s1_base, s1_last, ["mean_vv", "std_vv"], "s1_delta"),
        (s2_base, s2_last, ["median_ndvi", "median_ndmi"], "s2_delta"),
    ]:
        for k in keys:
            vb = (stats_base_d or {}).get(k, np.zeros((h, w), dtype=np.float32))
            vl = (stats_last_d or {}).get(k, np.zeros((h, w), dtype=np.float32))
            _append_2d((vl - vb).astype(np.float32), f"{prefix}_{k}")

    if s2_multi:
        if "s2_worst_drop" in s2_multi:
            _append_2d(s2_multi["s2_worst_drop"], "s2_worst_drop")
        if "s2_year_of_drop" in s2_multi:
            _append_2d(s2_multi["s2_year_of_drop"].astype(np.float32), "s2_year_of_drop")

    if terrain:
        for k in ("elevation", "slope_deg"):
            if k in terrain:
                _append_2d(terrain[k].astype(np.float32), f"terrain_{k}")

    X = np.concatenate(layers, axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if X.shape != (n, len(names)):
        raise AssertionError(f"pack_features: shape {X.shape} vs {len(names)} names")
    return X, names
