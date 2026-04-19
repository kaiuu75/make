"""Raster alignment / reprojection onto a reference grid.

The Sentinel-2 monthly composite grid (UTM, 10 m) is the canonical
reference. AEF ships on a lat/lon grid, Sentinel-1 at 30 m, and weak labels
on assorted grids — every feature and label raster is reprojected onto the
chosen S2 grid before stacking or fusion.
"""

from __future__ import annotations

import numpy as np
from rasterio.warp import Resampling, reproject


def reproject_to_grid(
    src_array: np.ndarray,
    *,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.nearest,
    src_nodata=None,
    dst_nodata=None,
    dtype=None,
) -> np.ndarray:
    """Reproject a single-band ``(H, W)`` raster onto a destination grid."""
    if src_array.ndim != 2:
        raise ValueError(f"reproject_to_grid expects 2-D input, got {src_array.shape}")
    out_dtype = np.dtype(dtype) if dtype is not None else src_array.dtype
    dst = np.zeros(dst_shape, dtype=out_dtype)
    if dst_nodata is not None:
        dst.fill(dst_nodata)
    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst


def reproject_multiband_to_grid(
    src_array: np.ndarray,
    *,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
    src_nodata=None,
    dst_nodata=None,
) -> np.ndarray:
    """Reproject a multi-band ``(C, H, W)`` raster onto a destination grid."""
    if src_array.ndim != 3:
        raise ValueError(
            f"reproject_multiband_to_grid expects (C,H,W), got {src_array.shape}"
        )
    c = src_array.shape[0]
    out = np.zeros((c, dst_shape[0], dst_shape[1]), dtype=src_array.dtype)
    for i in range(c):
        out[i] = reproject_to_grid(
            src_array[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_shape=dst_shape,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )
    return out
