"""Light-weight smoke tests that do not require the full challenge dataset.

Run with::

    pytest tests/ -q

or directly with ``python tests/test_smoke.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deforest2.data.paths import region_of  # noqa: E402
from deforest2.features.aef import aef_features  # noqa: E402
from deforest2.features.satellite import pack_features  # noqa: E402
from deforest2.features.terrain import (  # noqa: E402
    _slope_degrees,
    copernicus_dem_urls_for_bounds,
)
from deforest2.labels.fusion import fuse  # noqa: E402
from deforest2.labels.parsers import WeakLabel  # noqa: E402
from deforest2.models.gbm import (  # noqa: E402
    compute_positive_weights,
    select_candidate_pixels,
    subsample_positives_stratified,
)


def test_region_of() -> None:
    assert region_of("18NWG_6_6") == "18NWG"
    assert region_of("47QMB_0_8") == "47QMB"


def test_aef_features_shapes() -> None:
    rng = np.random.default_rng(0)
    aef = {y: rng.standard_normal((8, 16, 16)).astype(np.float32) for y in (2020, 2021, 2024)}
    f = aef_features(aef, multi_year_drift=True)
    assert f["aef_base"].shape == (8, 16, 16)
    assert f["aef_delta"].shape == (8, 16, 16)
    assert f["aef_norm"].shape == (16, 16)
    assert "aef_max_drift" in f and f["aef_max_drift"].shape == (16, 16)
    assert "aef_year_of_drift" in f


def test_pack_features_dense() -> None:
    rng = np.random.default_rng(0)
    aef = {y: rng.standard_normal((8, 8, 8)).astype(np.float32) for y in (2020, 2025)}
    f = aef_features(aef, multi_year_drift=True)
    X, names = pack_features(f, s2_base=None, s2_last=None, s1_base=None, s1_last=None)
    assert X.shape == (8 * 8, len(names))
    assert not np.isnan(X).any()


def test_fuse_basic() -> None:
    h, w = 4, 4
    a = np.zeros((h, w), dtype=np.float32)
    d = np.zeros((h, w), dtype=np.int32)
    a[0, 0] = 0.95
    a[1, 1] = 0.55
    b = np.zeros((h, w), dtype=np.float32)
    b[1, 1] = 0.80
    d2 = np.zeros((h, w), dtype=np.int32)
    wl_a = WeakLabel(a, d)
    wl_b = WeakLabel(b, d2)
    fused = fuse({"src_a": wl_a, "src_b": wl_b},
                 agreement_threshold=0.7, single_threshold=0.9)
    assert fused.binary[0, 0] == 1                  # single high
    assert fused.binary[1, 1] == 1                  # two agreeing at >= 0.7
    assert fused.max_confidence[0, 0] == 0.95
    assert fused.max_confidence[1, 1] == 0.80
    assert (fused.binary.sum()) == 2


def test_select_candidate_pixels_ignore_band() -> None:
    h, w = 4, 4
    binary = np.zeros((h, w), dtype=np.uint8)
    max_conf = np.zeros((h, w), dtype=np.float32)
    agree = np.zeros((h, w), dtype=np.uint8)
    days = np.zeros((h, w), dtype=np.int32)
    forest = np.ones((h, w), dtype=bool)
    binary[0, 0] = 1
    max_conf[0, 0] = 0.95
    max_conf[0, 1] = 0.5                            # ignored
    max_conf[0, 2] = 0.1                            # confident negative
    pos_idx, neg_idx, pos_conf, pos_agree, _ = select_candidate_pixels(
        binary, max_conf, agree, days, forest, hard_negative_max=0.2
    )
    assert pos_idx.size == 1
    assert pos_conf[0] == 0.95
    # 15 non-positive pixels, only one is < 0.2 so one confident negative.
    # But max_conf is zero for the rest, so they're all valid negatives.
    # => pixel (0,1) must NOT appear in neg_idx because 0.5 > 0.2.
    assert pos_idx[0] not in set(neg_idx.tolist())
    assert (0 * w + 1) not in set(neg_idx.tolist())
    assert (0 * w + 2) in set(neg_idx.tolist())


def test_stratified_positives_returns_positions() -> None:
    rng = np.random.default_rng(0)
    pos_conf = rng.uniform(0.5, 1.0, size=1_000).astype(np.float32)
    idx = subsample_positives_stratified(pos_conf, cap=200, n_strata=4, rng=rng)
    assert idx.dtype.kind == "i"
    assert 0 <= idx.min() and idx.max() < pos_conf.size
    assert idx.size == 200
    # Rough balance across strata.
    edges = np.quantile(pos_conf, [0.25, 0.5, 0.75])
    buckets = np.digitize(pos_conf[idx], edges)
    counts = np.bincount(buckets, minlength=4)
    assert counts.min() >= 30


def test_compute_positive_weights() -> None:
    conf = np.array([0.5, 1.0], dtype=np.float32)
    agree = np.array([1, 3], dtype=np.float32)
    w = compute_positive_weights(conf, agree)
    np.testing.assert_allclose(w, [0.5 * 1.25, 1.0 * 1.75])


def test_copernicus_dem_urls_for_bounds_single_tile() -> None:
    urls = copernicus_dem_urls_for_bounds(-75.8, 3.0, -75.7, 3.1)
    assert len(urls) == 1
    assert "Copernicus_DSM_COG_10_N03_00_W076_00_DEM" in urls[0]
    assert urls[0].startswith("https://copernicus-dem-30m.s3.amazonaws.com/")


def test_copernicus_dem_urls_for_bounds_crosses_1deg_boundary() -> None:
    urls = copernicus_dem_urls_for_bounds(104.9, 13.9, 105.1, 14.1)
    assert len(urls) == 4
    names = "\n".join(urls)
    for tag in ("N13_00_E104", "N13_00_E105", "N14_00_E104", "N14_00_E105"):
        assert tag in names


def test_slope_degrees_monotonic_gradient() -> None:
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    i, j = np.indices((64, 64), dtype=np.float32)
    dem = (i * 10.0 + j * 0.0).astype(np.float32)     # ~1 m rise per 30 m row
    transform = from_origin(500_000.0, 4_000_000.0, 30.0, 30.0)
    slope = _slope_degrees(dem, transform, CRS.from_epsg(32618))
    assert slope.shape == dem.shape
    assert np.all(slope >= 0.0)
    assert np.all(slope < 90.0)
    assert slope[10, 10] > 10.0
    flat = np.ones((8, 8), dtype=np.float32) * 123.4
    flat_slope = _slope_degrees(flat, transform, CRS.from_epsg(32618))
    assert float(flat_slope.max()) == 0.0


def test_pack_features_with_terrain() -> None:
    rng = np.random.default_rng(0)
    aef = {y: rng.standard_normal((4, 8, 8)).astype(np.float32) for y in (2020, 2025)}
    f = aef_features(aef, multi_year_drift=True)
    terr = {
        "elevation": rng.standard_normal((8, 8)).astype(np.float32),
        "slope_deg": rng.uniform(0.0, 45.0, size=(8, 8)).astype(np.float32),
    }
    X, names = pack_features(
        f, s2_base=None, s2_last=None, s1_base=None, s1_last=None, terrain=terr
    )
    assert "terrain_elevation" in names
    assert "terrain_slope_deg" in names
    assert X.shape == (8 * 8, len(names))
    assert not np.isnan(X).any()


if __name__ == "__main__":
    test_region_of()
    test_aef_features_shapes()
    test_pack_features_dense()
    test_fuse_basic()
    test_select_candidate_pixels_ignore_band()
    test_stratified_positives_returns_positions()
    test_compute_positive_weights()
    test_copernicus_dem_urls_for_bounds_single_tile()
    test_copernicus_dem_urls_for_bounds_crosses_1deg_boundary()
    test_slope_degrees_monotonic_gradient()
    test_pack_features_with_terrain()
    print("smoke tests OK")
