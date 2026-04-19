"""Raster readers for the challenge modalities.

Each reader returns ``(array, profile)`` where ``profile`` is a small dict
with ``transform``, ``crs``, ``shape`` and ``dtype``. We deliberately do not
propagate full rasterio profile objects through the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio


_S1_FILE_RE = re.compile(
    r".*__s1_rtc_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<orbit>ascending|descending)\.tif$",
    re.IGNORECASE,
)
_S2_FILE_RE = re.compile(
    r".*__s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif$",
    re.IGNORECASE,
)


def _profile(src) -> dict:
    return {
        "transform": src.transform,
        "crs": src.crs,
        "shape": (src.height, src.width),
        "dtype": src.dtypes[0] if src.count else None,
    }


def read_aef(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read AEF embedding GeoTIFF as ``(C, H, W) float32`` + profile."""
    with rasterio.open(path) as src:
        data = src.read(out_dtype=np.float32)
        prof = _profile(src)
    return data, prof


def read_s2(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a Sentinel-2 L2A monthly composite as ``(B, H, W) float32``.

    Integer reflectance values are scaled by 1/10000.0 so NDVI math behaves
    as expected. ``0`` is kept as-is and is treated as nodata downstream.
    """
    with rasterio.open(path) as src:
        raw = src.read()
        prof = _profile(src)
    data = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        data = data / 10000.0
    return data, prof


def read_s1(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a Sentinel-1 RTC monthly composite as ``(H, W) float32`` in dB.

    Linear-power backscatter is converted to decibels because VV statistics
    are much more Gaussian in dB. Non-positive values become NaN.
    """
    with rasterio.open(path) as src:
        raw = src.read(1).astype(np.float32)
        prof = _profile(src)
    valid = raw > 0
    db = np.full(raw.shape, np.nan, dtype=np.float32)
    db[valid] = 10.0 * np.log10(raw[valid])
    return db, prof


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


def list_s1_months(s1_dir: str | Path) -> list[tuple[int, int, str]]:
    d = Path(s1_dir)
    if not d.is_dir():
        return []
    out: list[tuple[int, int, str]] = []
    for p in sorted(d.iterdir()):
        m = _S1_FILE_RE.match(p.name)
        if not m:
            continue
        out.append((int(m["year"]), int(m["month"]), m["orbit"].lower()))
    out.sort()
    return out


def list_s2_months(s2_dir: str | Path) -> list[tuple[int, int]]:
    d = Path(s2_dir)
    if not d.is_dir():
        return []
    out: list[tuple[int, int]] = []
    for p in sorted(d.iterdir()):
        m = _S2_FILE_RE.match(p.name)
        if not m:
            continue
        out.append((int(m["year"]), int(m["month"])))
    out.sort()
    return out
