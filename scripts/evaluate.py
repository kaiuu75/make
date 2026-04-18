"""Evaluate a submission GeoJSON against ground-truth polygons.

Ground truth for a tile can be built from the training weak-label consensus
so the same metrics we use on the hidden test set can be estimated locally.

Usage::

    # Build GT from the training consensus (treats it as pseudo-GT)
    python scripts/evaluate.py --config configs/default.yaml \
        --predictions submissions/baseline.geojson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deforest.config import load_config
from deforest.cli import _data_paths, _resolve_tiles
from deforest.evaluation.metrics import evaluate as eval_fc
from deforest.inference.tile_predict import predict_tile
from deforest.postprocess.polygonize import (
    merge_feature_collections,
    polygonize,
    write_geojson,
)


def build_pseudo_gt(cfg) -> dict:
    """Pseudo-GT = the consensus baseline over training tiles."""
    paths = _data_paths(cfg)
    tile_ids = _resolve_tiles(None, cfg, split="train")
    fcs: list[dict] = []
    for tid in tile_ids:
        pred = predict_tile(
            tid, paths, split="train", model="baseline",
            fusion_cfg={
                "agreement_threshold": cfg.raw["label_fusion"]["agreement_threshold"],
                "single_threshold": cfg.raw["label_fusion"]["single_threshold"],
            },
        )
        bcfg = cfg.raw["baseline"]
        fc = polygonize(
            pred.prob,
            transform=pred.transform,
            crs=pred.crs,
            threshold=0.5,
            min_area_ha=bcfg["min_area_ha"],
            morph_open_px=bcfg["morph_open_px"],
            morph_close_px=bcfg["morph_close_px"],
            time_step_raster=pred.time_step,
        )
        fcs.append(fc)
    return merge_feature_collections(fcs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to a GT geojson. If omitted, a pseudo-GT is built from the training consensus.",
    )
    p.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("submissions/_pseudo_gt.geojson"),
        help="Where to cache the pseudo-GT.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)

    if args.ground_truth is None:
        if args.gt_cache.exists():
            gt_fc = json.loads(args.gt_cache.read_text())
        else:
            print(f"[evaluate] building pseudo-GT from training consensus → {args.gt_cache}")
            gt_fc = build_pseudo_gt(cfg)
            write_geojson(gt_fc, args.gt_cache)
    else:
        gt_fc = json.loads(args.ground_truth.read_text())

    pred_fc = json.loads(args.predictions.read_text())
    res = eval_fc(pred_fc, gt_fc)
    print(json.dumps(res.as_dict(), indent=2))


if __name__ == "__main__":
    main()
