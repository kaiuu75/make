#!/usr/bin/env python3
"""Pre-compute per-tile feature + label caches under /mnt/scratch.

Usage (MI300X droplet):

    python scripts/preprocess_tiles.py \
        --config configs/server.yaml \
        --split train

Outputs are written under ``deep.cache_dir`` from the config (default
``/mnt/scratch/deforest/patches``). Parallelism is autoscaled to the vCPU
count; override with ``--workers N``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from deforest.config import load_config
from deforest.data.paths import DataPaths, discover_tiles, list_tiles
from deforest.preprocess import PreprocessConfig, preprocess_all
from deforest.runtime import detect_hardware


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/server.yaml"))
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--tiles", type=str, default=None, help="Comma-separated tile ids")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hw = detect_hardware()
    print(f"[preprocess] {hw.summary()}")

    data_cfg = cfg.raw["data"]
    paths = DataPaths(
        root=Path(data_cfg["root"]),
        s1_subdir=data_cfg["s1_subdir"],
        s2_subdir=data_cfg["s2_subdir"],
        aef_subdir=data_cfg["aef_subdir"],
        labels_subdir=data_cfg["labels_subdir"],
    )

    if args.tiles:
        tiles = [t.strip() for t in args.tiles.split(",") if t.strip()]
    else:
        key = "train_tiles_geojson" if args.split == "train" else "test_tiles_geojson"
        meta = Path(data_cfg.get(key, ""))
        tiles = list_tiles(meta) if meta.exists() else discover_tiles(paths.root, split=args.split)

    if not tiles:
        print(f"[preprocess] no tiles found for split={args.split}")
        return 1

    pre_cfg = cfg.raw.get("preprocess", {})
    deep_cfg = cfg.raw.get("deep", {})
    cache_dir = Path(args.cache_dir or deep_cfg.get("cache_dir") or hw.scratch_dir / "deforest/patches")

    p = PreprocessConfig(
        cache_dir=cache_dir,
        patch_size=int(deep_cfg.get("patch_size", 256)),
        patch_stride=int(pre_cfg.get("patch_stride", 192)),
        max_patches_per_tile=int(pre_cfg.get("max_patches_per_tile", 512)),
        forest_min_fraction=float(pre_cfg.get("forest_min_fraction", 0.05)),
        skip_if_empty=bool(pre_cfg.get("skip_if_empty", True)),
        month_calendar_start=tuple(cfg.raw.get("month_calendar", {}).get("start", "2020-01").split("-")[:2]),  # type: ignore
        months_in_calendar=int(cfg.raw.get("month_calendar", {}).get("months", 72)),
        fusion={
            "agreement_threshold": cfg.raw["label_fusion"]["agreement_threshold"],
            "single_threshold": cfg.raw["label_fusion"]["single_threshold"],
        },
        workers=args.workers or pre_cfg.get("workers"),
    )
    # Coerce month-start tuple of strings to tuple of ints
    y, m = p.month_calendar_start
    p.month_calendar_start = (int(y), int(m))

    print(f"[preprocess] tiles={len(tiles)} cache={p.cache_dir} workers={p.workers}")
    n_done = 0
    n_patches = 0

    def on_progress(tid: str, n: int) -> None:
        nonlocal n_done, n_patches
        n_done += 1
        n_patches += n
        print(f"[preprocess] {n_done}/{len(tiles)}  {tid}  patches={n}")

    preprocess_all(tiles, paths, p, split=args.split, on_progress=on_progress)
    print(f"[preprocess] done. {n_done} tiles, {n_patches} patches indexed at {p.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
