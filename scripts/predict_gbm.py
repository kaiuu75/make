"""Run a trained GBM over tiles and emit a submission GeoJSON.

The output file matches the format expected by the Makeathon leaderboard
(a single ``FeatureCollection`` in EPSG:4326, polygons ≥ 0.5 ha, optional
per-polygon ``time_step`` in YYMM). Postprocessing knobs are read from
``configs/default.yaml`` by default and can be overridden with a second
YAML file produced by ``scripts/tune_postprocess.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deforest2.config import load_config, merge_overrides  # noqa: E402
from deforest2.data.paths import DataPaths, discover_tiles, list_tiles  # noqa: E402
from deforest2.inference.tile_predict import predict_tile  # noqa: E402
from deforest2.models.gbm import PixelGBM  # noqa: E402
from deforest2.postprocess.polygonize import (  # noqa: E402
    merge_feature_collections,
    polygonize,
    write_geojson,
)

import yaml  # noqa: E402


def _resolve_tiles(tiles_arg: str | None, cfg, split: str) -> list[str]:
    if tiles_arg:
        return [t.strip() for t in tiles_arg.split(",") if t.strip()]
    data_cfg = cfg.raw["data"]
    key = "train_tiles_geojson" if split == "train" else "test_tiles_geojson"
    meta_path = Path(data_cfg.get(key, ""))
    if meta_path.exists():
        ids = list_tiles(meta_path)
        if ids:
            return ids
    return discover_tiles(Path(data_cfg["root"]), split=split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--gbm-model", type=Path, required=True)
    ap.add_argument("--postprocess", type=Path, default=None, help="Optional tuned postprocess YAML")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--tiles", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("submissions/submission.geojson"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    gbm_cfg = cfg.raw["gbm"]
    post = dict(
        threshold=float(gbm_cfg.get("prediction_threshold", 0.55)),
        morph_open_px=int(gbm_cfg.get("morph_open_px", 3)),
        morph_close_px=int(gbm_cfg.get("morph_close_px", 5)),
        min_area_ha=float(gbm_cfg.get("min_area_ha", 0.5)),
    )
    if args.postprocess and args.postprocess.exists():
        overrides = yaml.safe_load(args.postprocess.read_text()) or {}
        post = merge_overrides(post, overrides.get("postprocess", overrides))
        print(f"[predict] using tuned postprocess: {post}")

    data_cfg = cfg.raw["data"]
    paths = DataPaths(
        root=Path(data_cfg["root"]),
        s1_subdir=data_cfg.get("s1_subdir", "sentinel-1"),
        s2_subdir=data_cfg.get("s2_subdir", "sentinel-2"),
        aef_subdir=data_cfg.get("aef_subdir", "aef-embeddings"),
        labels_subdir=data_cfg.get("labels_subdir", "labels/train"),
    )
    tile_ids = _resolve_tiles(args.tiles, cfg, args.split)
    if not tile_ids:
        raise SystemExit(f"No tiles found for split={args.split}")

    gbm = PixelGBM().load(args.gbm_model)
    fusion_cfg = {
        "agreement_threshold": cfg.raw["label_fusion"]["agreement_threshold"],
        "single_threshold": cfg.raw["label_fusion"]["single_threshold"],
    }
    feature_cfg = cfg.raw.get("features", {}) or {}
    include_ts = bool(cfg.raw["submission"].get("include_time_step", True))

    fcs: list[dict] = []
    for tid in tqdm(tile_ids):
        pred = predict_tile(
            tid, paths, gbm,
            split=args.split,
            fusion_cfg=fusion_cfg,
            feature_cfg=feature_cfg,
        )
        fc = polygonize(
            pred.prob,
            transform=pred.transform,
            crs=pred.crs,
            threshold=float(post["threshold"]),
            min_area_ha=float(post["min_area_ha"]),
            morph_open_px=int(post["morph_open_px"]),
            morph_close_px=int(post["morph_close_px"]),
            time_step_raster=pred.time_step if include_ts else None,
        )
        fcs.append(fc)

    merged = merge_feature_collections(fcs)
    write_geojson(merged, args.out)
    print(f"[predict] wrote {len(merged['features'])} polygons → {args.out}")


if __name__ == "__main__":
    main()
