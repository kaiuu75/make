#!/usr/bin/env python3
"""Run ensemble inference (ChangeUNet + optional LightGBM) and write the
final GeoJSON submission.

Steps per tile:
1. Open the preprocessed cache (features + forest mask).
2. Run deep inference with Hann-window overlap blending.
3. If a LightGBM model is provided, run it on the same features.
4. Blend probabilities using configured weights.
5. Threshold → morphology → polygonize → (optional) time_step.
6. Accumulate a single FeatureCollection, write to disk.

Usage:
    python scripts/predict_ensemble.py \\
        --config configs/server.yaml \\
        --deep-ckpt /mnt/scratch/deforest/checkpoints/best.pt \\
        --gbm-model models/gbm.txt \\
        --split test \\
        --out submissions/submission.geojson
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deforest.config import load_config
from deforest.deep.dataset import CachedTile
from deforest.deep.predict import autoscale_infer, load_checkpoint, month_idx_to_yymm, predict_tile
from deforest.ensemble import EnsembleWeights, blend
from deforest.features.satellite import pack_features  # noqa: F401 (keeps order stable)
from deforest.models.gbm import PixelGBM
from deforest.postprocess.polygonize import merge_feature_collections, polygonize, write_geojson
from deforest.runtime import detect_hardware
from rasterio import Affine
from rasterio.crs import CRS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/server.yaml"))
    ap.add_argument("--deep-ckpt", type=Path, default=None)
    ap.add_argument("--gbm-model", type=Path, default=None)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--tiles", type=str, default=None, help="Comma-separated tile ids")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.deep_ckpt is None and args.gbm_model is None:
        print("[predict_ensemble] need at least one of --deep-ckpt / --gbm-model")
        return 2

    cfg = load_config(args.config)
    hw = detect_hardware()
    print(f"[predict_ensemble] {hw.summary()}")

    deep_cfg = cfg.raw["deep"]
    ens_cfg = cfg.raw.get("ensemble", {})
    cache_dir = args.cache_dir or Path(deep_cfg.get("cache_dir") or hw.scratch_dir / "deforest/patches")

    # Determine tile list from the cache (each subdir == a tile).
    if args.tiles:
        tile_ids = [t.strip() for t in args.tiles.split(",") if t.strip()]
    else:
        tile_ids = sorted(
            p.name for p in cache_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()
        )

    if not tile_ids:
        print(f"[predict_ensemble] no tiles in {cache_dir}")
        return 1

    # --- Load models ----------------------------------------------------
    device, amp_dtype, batch_size = autoscale_infer(deep_cfg)
    deep_model = None
    if args.deep_ckpt is not None:
        deep_model, _ = load_checkpoint(args.deep_ckpt, device=device)
        print(f"[predict_ensemble] deep model loaded on {device} (bs={batch_size}, amp={amp_dtype})")

    gbm = None
    if args.gbm_model is not None:
        gbm = PixelGBM().load(args.gbm_model)
        print(f"[predict_ensemble] GBM loaded from {args.gbm_model}")

    weights = EnsembleWeights(
        deep=float(ens_cfg.get("weights", {}).get("deep", 0.7)),
        gbm=float(ens_cfg.get("weights", {}).get("gbm", 0.3)),
    )
    threshold = float(ens_cfg.get("threshold", deep_cfg.get("prediction_threshold", 0.5)))
    min_area = float(ens_cfg.get("min_area_ha", 0.5))
    open_px = int(ens_cfg.get("morph_open_px", 2))
    close_px = int(ens_cfg.get("morph_close_px", 5))

    cal_start = cfg.raw.get("month_calendar", {}).get("start", "2020-01")
    start_year, start_month = [int(v) for v in cal_start.split("-")[:2]]

    fcs: list[dict] = []
    for tid in tile_ids:
        print(f"[predict_ensemble] tile {tid}")
        tile = CachedTile.open(cache_dir / tid)
        transform = Affine(*tile.meta["transform"][:6])
        crs = CRS.from_string(tile.meta["crs"])

        # ---- deep ----
        prob_deep = None
        month_yymm = None
        if deep_model is not None:
            prob_deep, expected_idx = predict_tile(
                tile,
                deep_model,
                patch_size=int(deep_cfg.get("patch_size", 256)),
                overlap=int(deep_cfg.get("overlap", 64)),
                batch_size=batch_size,
                amp_dtype=amp_dtype,
                device=device,
            )
            month_yymm = month_idx_to_yymm(expected_idx, start_year, start_month)
            prob_deep *= np.asarray(tile.forest, dtype=np.float32)

        # ---- GBM ----
        prob_gbm = None
        if gbm is not None:
            F, H, W = tile.features.shape
            X = np.asarray(tile.features, dtype=np.float32).reshape(F, H * W).T
            prob_gbm = gbm.predict_proba(X).reshape(H, W).astype(np.float32)
            prob_gbm *= np.asarray(tile.forest, dtype=np.float32)

        prob = blend(prob_deep, prob_gbm, weights)
        if month_yymm is None:
            month_yymm = np.zeros(prob.shape, dtype=np.int32)

        fc = polygonize(
            prob,
            transform=transform,
            crs=crs,
            threshold=threshold,
            min_area_ha=min_area,
            morph_open_px=open_px,
            morph_close_px=close_px,
            time_step_raster=month_yymm if cfg.raw["submission"]["include_time_step"] else None,
        )
        fcs.append(fc)

    merged = merge_feature_collections(fcs)
    write_geojson(merged, args.out)
    print(f"[predict_ensemble] wrote {len(merged['features'])} polygons → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
