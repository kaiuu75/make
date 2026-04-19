"""Grid-search ``(threshold, morph_open_px, morph_close_px, min_area_ha)``
against the held-out validation tiles to directly optimise polygon Union IoU.

The target ground-truth is built from the fused weak labels of the
validation tiles — it is not real ground truth, but it is the best proxy we
have without a labelled hold-out. Because the *same* fusion + forest mask
is used to define the "ground truth" as to train the model, use the tuner's
output as a *relative* ranking signal, not an absolute accuracy number.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deforest2.config import load_config  # noqa: E402
from deforest2.data.paths import DataPaths, list_tiles, region_of  # noqa: E402
from deforest2.inference.tile_predict import predict_tile  # noqa: E402
from deforest2.models.gbm import PixelGBM  # noqa: E402
from deforest2.postprocess.metrics import (  # noqa: E402
    evaluate,
    fused_labels_to_feature_collection,
)
from deforest2.postprocess.polygonize import polygonize  # noqa: E402


DEFAULT_GRID = {
    "threshold": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "morph_open_px": [0, 1, 3],
    "morph_close_px": [0, 3, 5, 9],
    "min_area_ha": [0.3, 0.5, 1.0, 2.0],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--gbm-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("configs/tuned_postprocess.yaml"))
    ap.add_argument(
        "--val-tiles", type=str, default=None,
        help="Comma-separated tile ids to use as val set (default: gbm.val_regions).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.raw["data"]
    gbm_cfg = cfg.raw["gbm"]
    paths = DataPaths(
        root=Path(data_cfg["root"]),
        s1_subdir=data_cfg.get("s1_subdir", "sentinel-1"),
        s2_subdir=data_cfg.get("s2_subdir", "sentinel-2"),
        aef_subdir=data_cfg.get("aef_subdir", "aef-embeddings"),
        labels_subdir=data_cfg.get("labels_subdir", "labels/train"),
    )

    if args.val_tiles:
        tile_ids = [t.strip() for t in args.val_tiles.split(",") if t.strip()]
    else:
        prefix_chars = int(gbm_cfg.get("region_prefix_chars", 5))
        val_regions = gbm_cfg.get("val_regions", []) or []
        all_ids = list_tiles(Path(data_cfg["train_tiles_geojson"]))
        tile_ids = [
            tid for tid in all_ids
            if region_of(tid, prefix_chars=prefix_chars) in val_regions
        ]
    if not tile_ids:
        raise SystemExit("No validation tiles available; set --val-tiles or gbm.val_regions.")

    print(f"[tune] using val tiles: {tile_ids}")
    gbm = PixelGBM().load(args.gbm_model)

    fusion_cfg = {
        "agreement_threshold": cfg.raw["label_fusion"]["agreement_threshold"],
        "single_threshold": cfg.raw["label_fusion"]["single_threshold"],
    }
    feature_cfg = cfg.raw.get("features", {}) or {}

    cached: list[dict] = []
    for tid in tqdm(tile_ids, desc="predict val"):
        pred = predict_tile(
            tid, paths, gbm,
            split="train",
            fusion_cfg=fusion_cfg,
            feature_cfg=feature_cfg,
        )
        if pred.fused is None:
            continue
        gt_fc = fused_labels_to_feature_collection(
            pred.fused.binary, transform=pred.transform, crs=pred.crs,
            min_area_ha=float(gbm_cfg.get("min_area_ha", 0.5)),
        )
        cached.append({"pred": pred, "gt_fc": gt_fc})

    if not cached:
        raise SystemExit("No validation tiles produced usable fused labels.")

    # Grid search.
    combos = list(
        itertools.product(
            DEFAULT_GRID["threshold"],
            DEFAULT_GRID["morph_open_px"],
            DEFAULT_GRID["morph_close_px"],
            DEFAULT_GRID["min_area_ha"],
        )
    )
    print(f"[tune] grid: {len(combos)} combos over {len(cached)} tiles")

    records = []
    best = None
    for (thr, mo, mc, area) in tqdm(combos, desc="grid"):
        ious = []
        recalls = []
        fprs = []
        for entry in cached:
            pred = entry["pred"]
            fc = polygonize(
                pred.prob,
                transform=pred.transform,
                crs=pred.crs,
                threshold=float(thr),
                min_area_ha=float(area),
                morph_open_px=int(mo),
                morph_close_px=int(mc),
            )
            result = evaluate(fc, entry["gt_fc"])
            ious.append(result.union_iou)
            recalls.append(result.polygon_recall)
            fprs.append(result.polygon_fpr_proxy)
        mean_iou = float(np.mean(ious))
        rec = {
            "threshold": thr,
            "morph_open_px": mo,
            "morph_close_px": mc,
            "min_area_ha": area,
            "mean_iou": mean_iou,
            "mean_recall": float(np.mean(recalls)),
            "mean_fpr_proxy": float(np.mean(fprs)),
        }
        records.append(rec)
        if best is None or mean_iou > best["mean_iou"]:
            best = rec

    assert best is not None
    print(f"[tune] best: {best}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        yaml.safe_dump({"postprocess": {
            "threshold": float(best["threshold"]),
            "morph_open_px": int(best["morph_open_px"]),
            "morph_close_px": int(best["morph_close_px"]),
            "min_area_ha": float(best["min_area_ha"]),
        }}, f, sort_keys=False)
    report_path = args.out.with_suffix(args.out.suffix + ".report.json")
    with report_path.open("w") as f:
        json.dump({"best": best, "all": records}, f, indent=2)
    print(f"[tune] wrote {args.out} (full report → {report_path.name})")


if __name__ == "__main__":
    main()
