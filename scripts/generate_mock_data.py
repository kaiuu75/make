"""Generate a synthetic tile that matches the challenge dataset layout.

Used so that the whole pipeline can run on a MacBook before the real S3
dataset has been downloaded.

Example::

    python scripts/generate_mock_data.py --out data/makeathon-challenge --tile MOCK_0_0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deforest.data.mock import generate_mock_tile


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/makeathon-challenge"))
    p.add_argument("--tile", type=str, default="MOCK_0_0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    generate_mock_tile(args.out, tile_id=args.tile, seed=args.seed)
    print(f"Wrote mock tile '{args.tile}' under {args.out}")


if __name__ == "__main__":
    main()
