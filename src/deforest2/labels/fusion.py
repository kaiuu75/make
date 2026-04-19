"""Weighted consensus fusion of RADD / GLAD-L / GLAD-S2 weak labels.

Given ``{name: WeakLabel}`` sources aligned to a common grid, produce a
:class:`FusedLabels` carrying per-pixel

* ``binary``              — 1 if the pixel is deforestation per the rules, else 0.
* ``confidence``          — max confidence across sources *on positive pixels*
  (0 elsewhere), float32 in ``[0, 1]``.
* ``max_confidence``      — the raw max confidence across sources, **without**
  the positive mask, also in ``[0, 1]``. This is what :func:`subsample_pixels`
  uses to pick *confident negatives* and *ignore* the ambiguous middle band.
* ``agree_count``         — number of sources flagging the pixel (0..3).
* ``median_days``         — median UNIX day across agreeing sources (0 elsewhere).

Fusion rule (defaults in ``configs/default.yaml``)::

    positive  ⇔  (max_conf ≥ agreement_threshold AND agree_count ≥ 2)
              OR (max_conf ≥ single_threshold)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parsers import WeakLabel


@dataclass
class FusedLabels:
    binary: np.ndarray          # (H, W) uint8
    confidence: np.ndarray      # (H, W) float32, 0 outside binary positives
    max_confidence: np.ndarray  # (H, W) float32, raw max across sources
    agree_count: np.ndarray     # (H, W) uint8
    median_days: np.ndarray     # (H, W) int32, UNIX days


def fuse(
    sources: dict[str, WeakLabel],
    *,
    agreement_threshold: float = 0.7,
    single_threshold: float = 0.9,
    forest_mask_2020: np.ndarray | None = None,
) -> FusedLabels:
    """Combine weak label sources into a single labelled raster."""
    if not sources:
        raise ValueError("fuse() requires at least one WeakLabel source")

    shapes = {w.confidence.shape for w in sources.values()}
    if len(shapes) != 1:
        raise ValueError(f"All sources must share one shape, got {shapes}")

    confs = np.stack([s.confidence for s in sources.values()], axis=0)   # (S, H, W)
    days_stack = np.stack([s.days for s in sources.values()], axis=0)    # (S, H, W)

    max_conf = confs.max(axis=0).astype(np.float32)
    flag = confs > 0
    agree = flag.sum(axis=0).astype(np.uint8)

    positive = (
        (max_conf >= agreement_threshold) & (agree >= 2)
    ) | (max_conf >= single_threshold)

    if forest_mask_2020 is not None:
        positive &= forest_mask_2020.astype(bool)

    median_days = _masked_median(days_stack, flag).astype(np.int32)
    median_days[~positive] = 0

    binary = positive.astype(np.uint8)
    return FusedLabels(
        binary=binary,
        confidence=(max_conf * binary).astype(np.float32),
        max_confidence=max_conf,
        agree_count=agree,
        median_days=median_days,
    )


def _masked_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median of ``values`` along axis 0 where ``mask`` is true, 0 otherwise."""
    import warnings

    v = np.where(mask, values.astype(np.float64), np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(v, axis=0)
    med = np.where(np.isnan(med), 0, med)
    return med
