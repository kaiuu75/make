"""Leave-one-region-out cross-validation for the GBM (Improvement 5).

For each MGRS-prefix region present in the training tiles, we

1. build a config override with that region as the validation set,
2. shell out to ``scripts/train_gbm.py`` to train a fold-specific GBM,
3. call :func:`predict_tile` on the held-out tiles and evaluate polygon
   Union IoU against the fused weak labels.

Outputs a per-region report to ``models/cv_report.json`` so you can see
how the model generalises across regions — the key signal for
cross-continent performance (Africa vs Amazon, etc.).

This script is intentionally subprocess-based: LightGBM internally holds
references to the training dataset that are easier to free between folds
by spawning a fresh process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

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


def _write_fold_config(base_cfg_path: Path, val_region: str, tmp_dir: Path) -> Path:
    cfg = load_config(base_cfg_path)
    data = dict(cfg.raw)
    data.setdefault("gbm", {})
    data["gbm"]["val_regions"] = [val_region]
    p = tmp_dir / f"fold_{val_region}.yaml"
    with p.open("w") as f:
        yaml.safe_dump(dict(data), f)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out", type=Path, default=Path("models/cv_report.json"))
    ap.add_argument("--min-region-tiles", type=int, default=1)
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
    prefix_chars = int(gbm_cfg.get("region_prefix_chars", 5))
    meta_path = Path(data_cfg["train_tiles_geojson"])
    all_ids = list_tiles(meta_path)
    if not all_ids:
        raise SystemExit(f"No train tiles in {meta_path}")

    regions: dict[str, list[str]] = {}
    for tid in all_ids:
        r = region_of(tid, prefix_chars=prefix_chars)
        regions.setdefault(r, []).append(tid)
    fold_regions = [r for r, tiles in regions.items() if len(tiles) >= args.min_region_tiles]
    print(f"[cv] folding across {len(fold_regions)} regions")

    fusion_cfg = {
        "agreement_threshold": cfg.raw["label_fusion"]["agreement_threshold"],
        "single_threshold": cfg.raw["label_fusion"]["single_threshold"],
    }
    feature_cfg = cfg.raw.get("features", {}) or {}
    post = dict(
        threshold=float(gbm_cfg.get("prediction_threshold", 0.55)),
        morph_open_px=int(gbm_cfg.get("morph_open_px", 3)),
        morph_close_px=int(gbm_cfg.get("morph_close_px", 5)),
        min_area_ha=float(gbm_cfg.get("min_area_ha", 0.5)),
    )
    min_area = float(gbm_cfg.get("min_area_ha", 0.5))

    report: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for region in tqdm(fold_regions):
            fold_cfg = _write_fold_config(args.config, region, td)
            fold_model = td / f"gbm_{region}.txt"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "train_gbm.py"),
                "--config", str(fold_cfg),
                "--out", str(fold_model),
            ]
            print(f"[cv] training fold val_region={region}")
            res = subprocess.run(cmd, check=False)
            if res.returncode != 0 or not fold_model.exists():
                report[region] = {"error": f"training failed (rc={res.returncode})"}
                continue

            gbm = PixelGBM().load(fold_model)
            tile_results: list[dict] = []
            for tid in regions[region]:
                try:
                    pred = predict_tile(
                        tid, paths, gbm,
                        split="train",
                        fusion_cfg=fusion_cfg,
                        feature_cfg=feature_cfg,
                    )
                except FileNotFoundError as exc:
                    tile_results.append({"tile": tid, "error": str(exc)})
                    continue
                if pred.fused is None:
                    tile_results.append({"tile": tid, "error": "no weak labels"})
                    continue
                gt_fc = fused_labels_to_feature_collection(
                    pred.fused.binary, transform=pred.transform, crs=pred.crs,
                    min_area_ha=min_area,
                )
                fc = polygonize(
                    pred.prob,
                    transform=pred.transform, crs=pred.crs,
                    **{k: v for k, v in post.items()},
                )
                ev = evaluate(fc, gt_fc).as_dict()
                ev["tile"] = tid
                tile_results.append(ev)
            report[region] = {"tiles": tile_results}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[cv] wrote {args.out}")


if __name__ == "__main__":
    main()
