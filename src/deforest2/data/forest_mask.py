"""Build the 2020 forest mask used to restrict deforestation predictions.

The challenge defines a deforestation event as ``forest(2020) → non-forest``,
so predictions must be zeroed outside pixels that were forest in 2020.
Two fallback-capable strategies:

* :func:`forest_mask_from_aef` — cluster the 2020 AEF embedding into two
  groups via a cheap power-iteration PCA + sign split, then pick the side
  with the higher median NDVI as "forest".
* :func:`forest_mask_from_ndvi` — naive NDVI threshold, used when AEF is
  missing or clustering collapses.

Both return a boolean ``(H, W)`` where ``True`` = forest in 2020.
"""

from __future__ import annotations

import numpy as np


def forest_mask_from_ndvi(
    ndvi_median: np.ndarray,
    threshold: float = 0.55,
) -> np.ndarray:
    """``True`` where ``ndvi_median >= threshold`` (after NaN->0)."""
    arr = np.nan_to_num(ndvi_median, nan=0.0)
    return arr >= float(threshold)


def forest_mask_from_aef(
    aef_2020: np.ndarray,
    *,
    ndvi_2020_median: np.ndarray | None = None,
    ndvi_min: float = 0.55,
) -> np.ndarray:
    """Cluster AEF embeddings and return the high-NDVI cluster as forest."""
    if aef_2020.ndim != 3:
        raise ValueError(f"aef_2020 must be (C,H,W), got {aef_2020.shape}")
    c, h, w = aef_2020.shape
    flat = aef_2020.reshape(c, -1).T.astype(np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    mean = flat.mean(axis=0, keepdims=True)
    centred = flat - mean
    v = _power_iteration_pc1(centred, iters=12)
    scores = centred @ v

    pos = scores > 0
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        if ndvi_2020_median is not None:
            return forest_mask_from_ndvi(ndvi_2020_median, threshold=ndvi_min)
        return np.ones((h, w), dtype=bool)

    if ndvi_2020_median is not None:
        ndvi_flat = np.nan_to_num(ndvi_2020_median, nan=0.0).reshape(-1)
        ndvi_pos = float(np.median(ndvi_flat[pos]))
        ndvi_neg = float(np.median(ndvi_flat[neg]))
        forest_side = pos if ndvi_pos >= ndvi_neg else neg
    else:
        forest_side = pos if pos.sum() >= neg.sum() else neg

    mask_flat = forest_side.reshape(h, w)

    if ndvi_2020_median is not None:
        low_ndvi = np.nan_to_num(ndvi_2020_median, nan=0.0) < float(ndvi_min)
        mask_flat = mask_flat & ~low_ndvi

    return mask_flat.astype(bool)


def _power_iteration_pc1(x_centred: np.ndarray, iters: int = 10) -> np.ndarray:
    """Top right-singular vector of ``x_centred`` (N, C) via power iteration."""
    c = x_centred.shape[1]
    rng = np.random.default_rng(0)
    v = rng.standard_normal(c).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(max(1, int(iters))):
        xv = x_centred @ v
        v = x_centred.T @ xv
        v /= np.linalg.norm(v) + 1e-12
    return v
