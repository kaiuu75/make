"""Terrain features derived from the Copernicus GLO-30 DEM.

Adds two per-pixel features to the GBM:

* ``terrain_elevation`` — metres above WGS-84 ellipsoid (float32).
* ``terrain_slope_deg`` — degrees, computed on the native DEM grid and then
  reprojected onto the S2 reference grid.

Data source is the open Copernicus DEM S3 bucket (no auth)::

    https://copernicus-dem-30m.s3.amazonaws.com/
      Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM/
      Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM.tif

DEM tiles are 1° × 1°, named by their south-west integer corner. Each
Sentinel-2 reference tile (~10 km wide) needs between 1 and 4 DEM tiles.

Downloaded tiles are cached on disk. Subsequent runs are offline-friendly.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling, transform_bounds

from ..data.align import reproject_to_grid

logger = logging.getLogger(__name__)


_DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"


# ---------------------------------------------------------------------------
# Tile-URL helpers
# ---------------------------------------------------------------------------


def _dem_tile_name(lat: int, lon: int) -> str:
    """Return the Copernicus GLO-30 DEM tile name for an integer SW corner."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def _dem_tile_url(lat: int, lon: int) -> str:
    name = _dem_tile_name(lat, lon)
    return f"{_DEM_BUCKET}/{name}/{name}.tif"


def copernicus_dem_urls_for_bounds(
    w: float, s: float, e: float, n: float
) -> list[str]:
    """Enumerate the DEM-tile URLs covering the EPSG:4326 bounds ``(w, s, e, n)``."""
    if not (math.isfinite(w) and math.isfinite(s) and math.isfinite(e) and math.isfinite(n)):
        raise ValueError(f"non-finite bounds: {(w, s, e, n)}")

    lat_lo = int(math.floor(s))
    lat_hi = int(math.floor(n - 1e-9)) if n > lat_lo + 1 else lat_lo
    lat_hi = max(lat_hi, lat_lo)

    lon_lo = int(math.floor(w))
    lon_hi = int(math.floor(e - 1e-9)) if e > lon_lo + 1 else lon_lo
    lon_hi = max(lon_hi, lon_lo)

    urls: list[str] = []
    for lat in range(lat_lo, lat_hi + 1):
        for lon in range(lon_lo, lon_hi + 1):
            urls.append(_dem_tile_url(lat, lon))
    return urls


# ---------------------------------------------------------------------------
# Caching / fetching
# ---------------------------------------------------------------------------


def fetch_dem_to_cache(
    url: str,
    cache_dir: str | Path,
    *,
    timeout: float = 60.0,
) -> Path | None:
    """Download ``url`` into ``cache_dir`` if not already present.

    Returns the local path on success, or ``None`` if the download fails.
    Atomic: writes to a ``.part`` file and renames on completion.
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    out = cache_root / name
    if out.exists() and out.stat().st_size > 0:
        return out

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=name + ".", suffix=".part", dir=str(cache_root)
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, tmp_path.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp_path.replace(out)
        return out
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("DEM fetch failed for %s: %s", url, exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# Mosaic + slope
# ---------------------------------------------------------------------------


def read_mosaic_dem(
    urls: list[str],
    cache_dir: str | Path,
) -> tuple[np.ndarray, object, CRS] | None:
    """Download (if needed), open and mosaic DEM tiles. Returns ``(array, transform, crs)``.

    Returns ``None`` if no tile could be retrieved.
    """
    paths: list[Path] = []
    for url in urls:
        p = fetch_dem_to_cache(url, cache_dir)
        if p is not None:
            paths.append(p)
    if not paths:
        return None

    srcs = [rasterio.open(p) for p in paths]
    try:
        mosaic, transform = rio_merge(srcs)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()

    arr = mosaic[0].astype(np.float32)
    return arr, transform, crs


def _slope_degrees(dem: np.ndarray, transform, crs: CRS) -> np.ndarray:
    """Slope in degrees, computed on the native DEM grid."""
    if crs.is_geographic:
        # Approximate metric pixel size using the mid-latitude of the mosaic.
        # transform.a is degrees/pixel in x, transform.e is degrees/pixel in y.
        rows = dem.shape[0]
        mid_lat = transform.f + transform.e * (rows / 2.0)
        mid_lat_rad = math.radians(mid_lat)
        dx_m = abs(transform.a) * 111_320.0 * max(math.cos(mid_lat_rad), 1e-6)
        dy_m = abs(transform.e) * 110_540.0
    else:
        dx_m = abs(transform.a)
        dy_m = abs(transform.e)

    dem_f = np.nan_to_num(dem, nan=0.0).astype(np.float32)
    gy, gx = np.gradient(dem_f, dy_m, dx_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    return slope


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _ref_lonlat_bounds(ref_transform, ref_crs, ref_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """EPSG:4326 bounds of the reference grid (w, s, e, n)."""
    h, w = ref_shape
    left = ref_transform.c
    top = ref_transform.f
    right = left + ref_transform.a * w
    bottom = top + ref_transform.e * h
    lo_w, lo_s, lo_e, lo_n = transform_bounds(
        ref_crs, "EPSG:4326",
        left=left, bottom=bottom, right=right, top=top,
        densify_pts=21,
    )
    return lo_w, lo_s, lo_e, lo_n


def terrain_features(
    ref_transform,
    ref_crs,
    ref_shape: tuple[int, int],
    *,
    cache_dir: str | Path,
) -> dict[str, np.ndarray] | None:
    """Build ``terrain_elevation`` + ``terrain_slope_deg`` on the reference grid.

    Returns ``None`` on network / data failures so callers can continue
    gracefully without terrain features.
    """
    try:
        w, s, e, n = _ref_lonlat_bounds(ref_transform, ref_crs, ref_shape)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("terrain_features: bounds reprojection failed: %s", exc)
        return None

    try:
        urls = copernicus_dem_urls_for_bounds(w, s, e, n)
    except ValueError as exc:
        logger.warning("terrain_features: bad bounds %s: %s", (w, s, e, n), exc)
        return None
    if not urls:
        return None

    mosaic = read_mosaic_dem(urls, cache_dir)
    if mosaic is None:
        logger.warning("terrain_features: no DEM tiles available for bounds %s", (w, s, e, n))
        return None
    dem_arr, dem_transform, dem_crs = mosaic

    slope = _slope_degrees(dem_arr, dem_transform, dem_crs)

    elev_on_ref = reproject_to_grid(
        dem_arr,
        src_transform=dem_transform, src_crs=dem_crs,
        dst_transform=ref_transform, dst_crs=ref_crs,
        dst_shape=ref_shape,
        resampling=Resampling.bilinear,
        dtype=np.float32,
    )
    slope_on_ref = reproject_to_grid(
        slope,
        src_transform=dem_transform, src_crs=dem_crs,
        dst_transform=ref_transform, dst_crs=ref_crs,
        dst_shape=ref_shape,
        resampling=Resampling.bilinear,
        dtype=np.float32,
    )

    return {
        "elevation": np.nan_to_num(elev_on_ref, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "slope_deg": np.nan_to_num(slope_on_ref, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    }
