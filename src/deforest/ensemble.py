"""Blend deep-model and LightGBM probabilities, then polygonize once.

At inference time we get, per test tile:

* ``prob_deep``  — float32 raster from :func:`deep.predict.predict_tile`
* ``prob_gbm``   — float32 raster from :class:`deforest.models.gbm.PixelGBM`

The ensemble rule is a weighted average followed by the usual morphology +
area filter. We treat the deep model as the stronger signal (weight 0.7 by
default) but keep a non-trivial GBM contribution — the two error profiles
are complementary: the deep model tends to over-smooth small polygons, the
GBM preserves them because it is pixel-independent and trained directly on
weak labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EnsembleWeights:
    deep: float = 0.70
    gbm: float = 0.30

    def normalized(self) -> "EnsembleWeights":
        s = self.deep + self.gbm
        if s <= 0:
            return EnsembleWeights(1.0, 0.0)
        return EnsembleWeights(self.deep / s, self.gbm / s)


def blend(
    prob_deep: np.ndarray | None,
    prob_gbm: np.ndarray | None,
    weights: EnsembleWeights = EnsembleWeights(),
) -> np.ndarray:
    """Weighted average; falls back gracefully if one model is missing."""
    w = weights.normalized()
    if prob_deep is not None and prob_gbm is not None:
        if prob_deep.shape != prob_gbm.shape:
            raise ValueError(f"shape mismatch: {prob_deep.shape} vs {prob_gbm.shape}")
        return (w.deep * prob_deep + w.gbm * prob_gbm).astype(np.float32)
    if prob_deep is not None:
        return prob_deep.astype(np.float32)
    if prob_gbm is not None:
        return prob_gbm.astype(np.float32)
    raise ValueError("blend() needs at least one non-None prob raster")
