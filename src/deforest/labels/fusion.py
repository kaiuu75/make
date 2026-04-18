"""Weighted consensus fusion of RADD / GLAD-L / GLAD-S2 weak labels.

Given the three :class:`WeakLabel` sources aligned to a common grid, produce
four per-pixel rasters:

* ``binary_label``  — 1 if the pixel is deforestation per our rules, else 0.
* ``confidence``    — maximum confidence across sources that agree, in [0, 1].
* ``agree_count``   — how many sources flag the pixel (int, 0..3).
* ``median_days``   — the median alert date (UNIX days) across agreeing
  sources. 0 if no agreement.

Fusion rule (default from ``configs/default.yaml``)::

    positive  ⇔  (max_conf ≥ agreement_threshold AND agree_count ≥ 2)
              OR (max_conf ≥ single_threshold)

The exact thresholds and per-level scores are configurable so the same code
serves as (a) the Tier-0 zero-training baseline **and** (b) the target /
sample-weight generator for LightGBM training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parsers import WeakLabel


@dataclass
class FusedLabels:
    binary: np.ndarray     # (H, W) uint8
    confidence: np.ndarray  # (H, W) float32
    agree_count: np.ndarray  # (H, W) uint8
    median_days: np.ndarray  # (H, W) int32


def fuse(
    sources: dict[str, WeakLabel],
    *,
    agreement_threshold: float = 0.7,
    single_threshold: float = 0.9,
    forest_mask_2020: np.ndarray | None = None,
) -> FusedLabels:
    """Combine weak label sources into a single labelled raster.

    Parameters
    ----------
    sources
        Dict of ``{name: WeakLabel}`` — all arrays must share the same shape.
    agreement_threshold
        Min max-confidence needed when *multiple* sources agree.
    single_threshold
        Min confidence for a single-source positive call.
    forest_mask_2020
        Optional boolean raster of pixels that were forest in 2020. Positives
        outside this mask are demoted to 0.
    """
    if not sources:
        raise ValueError("fuse() requires at least one WeakLabel source")

    shapes = {w.confidence.shape for w in sources.values()}
    if len(shapes) != 1:
        raise ValueError(f"All sources must share one shape, got {shapes}")

    (h, w) = shapes.pop()
    confs = np.stack([s.confidence for s in sources.values()], axis=0)      # (S, H, W)
    days = np.stack([s.days for s in sources.values()], axis=0)             # (S, H, W)

    max_conf = confs.max(axis=0).astype(np.float32)
    flag = confs > 0
    agree = flag.sum(axis=0).astype(np.uint8)

    positive = ((max_conf >= agreement_threshold) & (agree >= 2)) | (max_conf >= single_threshold)

    if forest_mask_2020 is not None:
        positive &= forest_mask_2020.astype(bool)

    # Median date across only the agreeing sources. Use a masked reduction.
    median_days = _masked_median(days, flag).astype(np.int32)
    median_days[~positive] = 0

    binary = positive.astype(np.uint8)
    return FusedLabels(
        binary=binary,
        confidence=(max_conf * binary).astype(np.float32),
        agree_count=agree,
        median_days=median_days,
    )


def _masked_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median of ``values`` along axis 0 where ``mask`` is true, 0 otherwise."""
    import warnings

    v = np.where(mask, values.astype(np.float64), np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice"
        med = np.nanmedian(v, axis=0)
    med = np.where(np.isnan(med), 0, med)
    return med
