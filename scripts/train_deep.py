#!/usr/bin/env python3
"""Train :class:`ChangeUNet` on the precomputed patch cache.

Usage:
    python scripts/train_deep.py --config configs/server.yaml
Resume:
    python scripts/train_deep.py --config configs/server.yaml --resume /mnt/scratch/deforest/checkpoints/last.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from deforest.config import load_config
from deforest.deep.train import train
from deforest.runtime import detect_hardware


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/server.yaml"))
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--val-tiles", type=str, default=None, help="Comma-separated tile ids for validation")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hw = detect_hardware()
    deep = cfg.raw["deep"]

    cache_dir = args.cache_dir or Path(deep.get("cache_dir") or hw.scratch_dir / "deforest/patches")
    ckpt_dir = args.checkpoint_dir or Path(deep.get("checkpoint_dir") or hw.scratch_dir / "deforest/checkpoints")

    val_tiles = None
    if args.val_tiles:
        val_tiles = [t.strip() for t in args.val_tiles.split(",") if t.strip()]

    best = train(
        cache_dir=cache_dir,
        checkpoint_dir=ckpt_dir,
        cfg=deep,
        val_tiles=val_tiles,
        resume=args.resume,
    )
    print(f"[train_deep] best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
