import numpy as np
from affine import Affine
from rasterio.crs import CRS

from deforest.postprocess.polygonize import polygonize


def test_polygonize_empty_returns_empty_feature_collection():
    prob = np.zeros((32, 32), dtype=np.float32)
    transform = Affine(10, 0, 500_000, 0, -10, 10_000)
    crs = CRS.from_epsg(32618)
    fc = polygonize(prob, transform=transform, crs=crs, threshold=0.5, min_area_ha=0.0)
    assert fc["type"] == "FeatureCollection"
    assert fc["features"] == []


def test_polygonize_blob_produces_one_polygon():
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[20:40, 20:40] = 0.9  # 20x20 px × 10m = 200m × 200m = 4 ha
    transform = Affine(10, 0, 500_000, 0, -10, 10_000)
    crs = CRS.from_epsg(32618)
    fc = polygonize(prob, transform=transform, crs=crs, threshold=0.5, min_area_ha=0.5)
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    # Should be in EPSG:4326 after polygonize.
    x, y = feat["geometry"]["coordinates"][0][0]
    assert -180 <= x <= 180
    assert -90 <= y <= 90


def test_polygonize_area_filter_drops_tiny_blobs():
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[0:2, 0:2] = 0.9  # 2x2 px × 10m = 400 m² = 0.04 ha
    transform = Affine(10, 0, 500_000, 0, -10, 10_000)
    crs = CRS.from_epsg(32618)
    fc = polygonize(prob, transform=transform, crs=crs, threshold=0.5, min_area_ha=0.5)
    assert fc["features"] == []


def test_polygonize_adds_time_step_from_raster():
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[20:40, 20:40] = 0.9
    ts_raster = np.zeros((64, 64), dtype=np.int32)
    ts_raster[20:40, 20:40] = 2204
    transform = Affine(10, 0, 500_000, 0, -10, 10_000)
    crs = CRS.from_epsg(32618)
    fc = polygonize(
        prob,
        transform=transform, crs=crs,
        threshold=0.5, min_area_ha=0.5,
        time_step_raster=ts_raster,
    )
    assert len(fc["features"]) == 1
    assert fc["features"][0]["properties"]["time_step"] == 2204
