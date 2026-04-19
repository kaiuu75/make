# make2 — improved GBM pipeline for osapiens Makeathon 2026

Focused port of the LightGBM pixel-classifier from `make/` with seven targeted
improvements to raise Union-IoU on the hidden test set and improve
cross-region generalisation.

## What's inside

- A per-pixel LightGBM model trained on weak-label-fused targets from RADD,
  GLAD-L and GLAD-S2 (as described in the challenge notebook
  `makeathon26/challenge.ipynb`).
- A minimal inference pipeline that emits a submission-ready
  `FeatureCollection` GeoJSON in EPSG:4326 matching the format expected by
  `makeathon26/submission_utils.py`.
- A postprocessing tuner that grid-searches (threshold, morphology, area)
  on a held-out region to directly optimise polygon Union IoU.

## Improvements over `make/`

Each item maps to a weakness identified in the `make` GBM:

1. **Global stratified quota** in `scripts/train_gbm.py` replaces the old
   per-tile caps so one dominant tile cannot drown others out. Positives are
   additionally **confidence-stratified** across four buckets.
2. **Richer features** in `src/deforest2/features/satellite.py` and
   `features/aef.py`: per-year S2 stats now include p10/p90/std; all
   available years get an AEF trajectory drift (max + year-of-max) and an
   S2 worst-NDVI-drop (+ year-of-drop).
3. **Cross-entropy soft targets** and an **ignore band** for ambiguous
   label pixels. `src/deforest2/models/gbm.py` supports
   `objective="cross_entropy"`; `src/deforest2/labels/fusion.py` now
   returns a `hard_negative_mask` so `subsample_pixels` only sees confident
   negatives.
4. **Held-out validation + early stopping** — `train_gbm.py` splits tiles
   by MGRS prefix and forwards `eval_X/eval_y` to
   `PixelGBM.fit`; training terminates on validation log-loss plateau.
5. **Leave-one-region-out CV** via `scripts/cv_gbm.py` (runs a training
   fold per MGRS prefix and reports per-region polygon IoU).
6. **Postprocessing tuning**: `scripts/tune_postprocess.py` grid-searches
   `(threshold, morph_open_px, morph_close_px, min_area_ha)` on validation
   tiles and writes the winning tuple to `configs/tuned_postprocess.yaml`.
7. **Explicit class-weighting** — prevalence (`scale_pos_weight`),
   confidence trust (`w ∝ confidence · (1 + 0.25·agreement)`, normalised to
   unit mean per tile), and region balance are separate multiplicative
   factors rather than tangled together in a single sample-cap.

## Constraints taken from the Makeathon brief

(see `makeathon26/osapiens-challenge-full-description.md` and
`makeathon26/challenge.ipynb`)

- Submission = a single GeoJSON `FeatureCollection` in EPSG:4326.
- "Deforestation" means `forest(2020) → non-forest` **after 2020**.
- Polygons below **0.5 ha** are filtered out.
- Bonus: optional per-polygon `time_step` in YYMM.
- Training uses only the provided modalities (Sentinel-1, Sentinel-2,
  AlphaEarth embeddings) and the three weak-label sources.

## Install

```bash
cd make2
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

## Run

Assuming the makeathon dataset is at `../makeathon26/data/makeathon-challenge/`:

```bash
# train the GBM
python scripts/train_gbm.py --config configs/default.yaml --out models/gbm.txt

# tune postprocessing on the val tiles
python scripts/tune_postprocess.py --config configs/default.yaml \
       --gbm-model models/gbm.txt \
       --out configs/tuned_postprocess.yaml

# produce the submission
python scripts/predict_gbm.py --config configs/default.yaml \
       --gbm-model models/gbm.txt \
       --postprocess configs/tuned_postprocess.yaml \
       --split test --out submissions/submission.geojson

# leave-one-region-out CV (diagnostic)
python scripts/cv_gbm.py --config configs/default.yaml
```

## Layout

```
make2/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train_gbm.py
│   ├── predict_gbm.py
│   ├── tune_postprocess.py
│   └── cv_gbm.py
├── src/deforest2/
│   ├── config.py
│   ├── data/        (paths, readers, align, forest_mask)
│   ├── labels/      (parsers, fusion)
│   ├── features/    (aef, satellite)
│   ├── models/      (gbm)
│   ├── inference/   (tile_predict, time_step)
│   └── postprocess/ (polygonize)
└── tests/
    └── test_smoke.py
```
