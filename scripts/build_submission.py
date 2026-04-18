"""End-to-end submission builder.

Wraps ``deforest predict`` so ``make baseline`` / ``make submit`` work
without depending on the click entry point being installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deforest.cli import predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "gbm"], default="baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--tiles", default=None)
    parser.add_argument("--gbm-model", dest="gbm_model", type=Path, default=None)
    args = parser.parse_args()

    # Invoke the Click command in standalone mode.
    predict.callback(
        config=args.config,
        model=args.model,
        out_path=args.out,
        split=args.split,
        tiles=args.tiles,
        gbm_model=args.gbm_model,
    )


if __name__ == "__main__":
    main()
