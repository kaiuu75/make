# Architecture

```
make/
├── README.md                       ← how to run
├── requirements.txt                ← pinned deps
├── pyproject.toml                  ← package metadata
├── Makefile                        ← convenience targets
├── configs/
│   └── default.yaml                ← paths, thresholds, tile selection
├── docs/
│   ├── approach.md                 ← plan + paper analysis
│   └── architecture.md             ← this file
├── src/deforest/
│   ├── __init__.py
│   ├── config.py                   ← load YAML config → dataclass
│   ├── data/
│   │   ├── paths.py                ← tile-id ↔ file-path helpers
│   │   ├── readers.py              ← rasterio loaders (S1/S2/AEF)
│   │   ├── align.py                ← reproject + resample to common grid
│   │   ├── forest_mask.py          ← 2020 forest mask builder
│   │   └── mock.py                 ← synthetic tile generator for laptop dev
│   ├── labels/
│   │   ├── parsers.py              ← decode RADD / GLAD-L / GLAD-S2 encodings
│   │   └── fusion.py               ← weighted consensus → (mask, confidence, date)
│   ├── features/
│   │   ├── aef.py                  ← AEF(2020), AEF(last), Δ
│   │   └── satellite.py            ← S1 + S2 annual statistics, NDVI slope
│   ├── models/
│   │   ├── baseline.py             ← zero-training consensus model
│   │   ├── gbm.py                  ← LightGBM training + prediction
│   │   └── utae.py                 ← Tier 2 stub (date refinement)
│   ├── inference/
│   │   ├── tile_predict.py         ← run any model tile-by-tile
│   │   └── time_step.py            ← YYMM assignment per polygon
│   ├── postprocess/
│   │   └── polygonize.py           ← binary raster → polygons + time_step
│   ├── evaluation/
│   │   └── metrics.py              ← Union IoU, Polygon Recall, FPR, Year Acc
│   └── cli.py                      ← entrypoint
├── scripts/
│   ├── generate_mock_data.py       ← build a synthetic tile for dry-run
│   ├── build_submission.py         ← end-to-end: data → geojson
│   └── evaluate.py                 ← local metrics on training split
└── tests/
    ├── test_fusion.py
    ├── test_metrics.py
    └── test_polygonize.py
```

## Data contract

All models operate on a per‑tile dict:

```python
tile = {
    "tile_id": "18NWG_6_6",
    "crs": <rasterio CRS>,              # local UTM from Sentinel-2
    "transform": <affine.Affine>,       # S2 grid
    "shape": (H, W),
    "s2": np.ndarray | None,            # (T, 12, H, W) or annual stats
    "s1": np.ndarray | None,            # (T, 1, H, W)
    "aef": dict[int, np.ndarray],       # year → (64, H, W), reprojected to UTM
    "labels": {
        "radd":   (mask_HxW_uint8, date_HxW_int32, confidence_HxW_float32),
        "gladl":  (mask_HxW_uint8, date_HxW_int32, confidence_HxW_float32),
        "glads2": (mask_HxW_uint8, date_HxW_int32, confidence_HxW_float32),
    },
    "forest_mask_2020": np.ndarray,     # (H, W) bool
}
```

A `Predictor` is any callable `tile → (prob_HxW_float32, time_step_HxW_int32)`
where `time_step` is YYMM or 0 if unknown.

`postprocess.polygonize.to_geojson(prob, time_step, threshold, min_area_ha,
  transform, crs) → dict`
returns a submission‑ready FeatureCollection in EPSG:4326.
