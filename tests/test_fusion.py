import numpy as np

from deforest.labels.fusion import fuse
from deforest.labels.parsers import WeakLabel


def _wl(conf, days):
    return WeakLabel(
        confidence=np.asarray(conf, dtype=np.float32),
        days=np.asarray(days, dtype=np.int32),
    )


def test_single_high_confidence_is_positive():
    sources = {
        "radd": _wl([[1.0, 0.0]], [[18262, 0]]),        # 2020-01-01
        "gladl": _wl([[0.0, 0.0]], [[0, 0]]),
        "glads2": _wl([[0.0, 0.0]], [[0, 0]]),
    }
    out = fuse(sources, agreement_threshold=0.7, single_threshold=0.9)
    assert out.binary[0, 0] == 1
    assert out.binary[0, 1] == 0
    assert out.confidence[0, 0] == 1.0
    assert out.median_days[0, 0] == 18262


def test_agreement_with_two_sources():
    sources = {
        "radd": _wl([[0.6]], [[18262]]),
        "gladl": _wl([[0.9]], [[18300]]),
        "glads2": _wl([[0.0]], [[0]]),
    }
    out = fuse(sources, agreement_threshold=0.7, single_threshold=0.99)
    assert out.binary[0, 0] == 1
    # median of 2 days should be the average
    assert out.median_days[0, 0] in (18281, 18262, 18300)


def test_below_thresholds_is_negative():
    sources = {
        "radd": _wl([[0.6]], [[18262]]),
        "gladl": _wl([[0.0]], [[0]]),
        "glads2": _wl([[0.5]], [[18300]]),
    }
    out = fuse(sources, agreement_threshold=0.7, single_threshold=0.9)
    # Two sources agree but max confidence 0.6 < 0.7 → negative
    assert out.binary[0, 0] == 0


def test_forest_mask_demotes_outside_pixels():
    sources = {
        "radd": _wl([[1.0, 1.0]], [[18262, 18262]]),
        "gladl": _wl([[0.0, 0.0]], [[0, 0]]),
        "glads2": _wl([[0.0, 0.0]], [[0, 0]]),
    }
    mask = np.array([[True, False]])
    out = fuse(sources, forest_mask_2020=mask)
    assert out.binary[0, 0] == 1
    assert out.binary[0, 1] == 0
