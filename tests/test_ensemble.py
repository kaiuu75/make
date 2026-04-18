"""Ensemble blending is a tiny pure-numpy function — exercise the edges."""

from __future__ import annotations

import numpy as np
import pytest

from deforest.ensemble import EnsembleWeights, blend


def test_blend_weighted_average():
    a = np.ones((4, 4), dtype=np.float32) * 0.8
    b = np.ones((4, 4), dtype=np.float32) * 0.2
    p = blend(a, b, EnsembleWeights(deep=0.7, gbm=0.3))
    assert np.allclose(p, 0.7 * 0.8 + 0.3 * 0.2)


def test_blend_only_deep():
    a = np.full((3, 3), 0.5, dtype=np.float32)
    p = blend(a, None)
    np.testing.assert_array_equal(p, a)


def test_blend_only_gbm():
    b = np.full((3, 3), 0.25, dtype=np.float32)
    p = blend(None, b)
    np.testing.assert_array_equal(p, b)


def test_blend_rejects_shape_mismatch():
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        blend(a, b)


def test_blend_rejects_both_none():
    with pytest.raises(ValueError):
        blend(None, None)


def test_weights_normalize():
    w = EnsembleWeights(deep=2.0, gbm=2.0).normalized()
    assert w.deep == 0.5 and w.gbm == 0.5

    w = EnsembleWeights(deep=0.0, gbm=0.0).normalized()
    assert w.deep == 1.0 and w.gbm == 0.0
