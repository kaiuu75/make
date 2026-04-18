"""Command-line interface.

Entry point registered as ``deforest`` (see pyproject.toml). Commands:

* ``deforest mock`` — generate synthetic tile
* ``deforest predict`` — run a model over a set of tiles, write submission
* ``deforest evaluate`` — local metrics vs a ground-truth GeoJSON
* ``deforest runtime`` — print detected hardware + autoscaled defaults
  (useful for sanity-checking an MI300X droplet)
* ``deforest preprocess`` / ``deforest train-deep`` / ``deforest predict-ensemble``
  are thin wrappers around the equivalent scripts in ``scripts/`` so you can
  drive the server pipeline entirely through the ``deforest`` CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .config import load_config
from .data.mock import generate_mock_tile
from .data.paths import DataPaths, discover_tiles, list_tiles


@click.group()
def main():
    """osapiens Makeathon 2026 — deforestation CLI."""


@main.command()
@click.option("--out", "out_root", type=click.Path(path_type=Path), default=Path("data/makeathon-challenge"))
@click.option("--tile", "tile_id", type=str, default="MOCK_0_0")
def mock(out_root: Path, tile_id: str):
    """Generate a synthetic tile."""
    paths = generate_mock_tile(out_root, tile_id=tile_id)
    click.echo(f"mock tile '{tile_id}' written under {out_root}")
    for k, v in paths.items():
        click.echo(f"  {k}: {v}")


@main.command()
@click.option("--config", type=click.Path(path_type=Path, exists=True), default=Path("configs/default.yaml"))
@click.option("--model", type=click.Choice(["baseline", "gbm"]), default="baseline")
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
@click.option("--split", type=click.Choice(["train", "test"]), default="train")
@click.option("--tiles", type=str, default=None, help="Comma-separated tile ids; default = discover")
@click.option("--gbm-model", type=click.Path(path_type=Path), default=None, help="Path to saved LightGBM model")
def predict(
    config: Path,
    model: str,
    out_path: Path,
    split: str,
    tiles: str | None,
    gbm_model: Path | None,
):
    """Run a model over tiles and write one submission FeatureCollection."""
    cfg = load_config(config)
    paths = _data_paths(cfg)

    tile_ids = _resolve_tiles(tiles, cfg, split)
    if not tile_ids:
        click.echo(f"[deforest] no tiles found for split={split}", err=True)
        sys.exit(1)

    from .inference.tile_predict import predict_tile
    from .postprocess.polygonize import merge_feature_collections, polygonize, write_geojson

    loaded_gbm = None
    if model == "gbm":
        if gbm_model is None:
            click.echo("[deforest] --gbm-model is required when --model gbm", err=True)
            sys.exit(2)
        from .models.gbm import PixelGBM

        loaded_gbm = PixelGBM().load(gbm_model)

    fcs: list[dict] = []
    for tid in tile_ids:
        click.echo(f"[deforest] predicting {tid}")
        pred = predict_tile(
            tid,
            paths,
            split=split,
            model=model,
            gbm=loaded_gbm,
            fusion_cfg={
                "agreement_threshold": cfg.raw["label_fusion"]["agreement_threshold"],
                "single_threshold": cfg.raw["label_fusion"]["single_threshold"],
            },
        )
        bcfg = cfg.raw["baseline" if model == "baseline" else "gbm"]
        threshold = 0.5 if model == "baseline" else bcfg.get("prediction_threshold", 0.55)
        fc = polygonize(
            pred.prob,
            transform=pred.transform,
            crs=pred.crs,
            threshold=threshold,
            min_area_ha=bcfg.get("min_area_ha", 0.5),
            morph_open_px=bcfg.get("morph_open_px", 0),
            morph_close_px=bcfg.get("morph_close_px", 0),
            time_step_raster=pred.time_step if cfg.raw["submission"]["include_time_step"] else None,
        )
        fcs.append(fc)

    merged = merge_feature_collections(fcs)
    write_geojson(merged, out_path)
    click.echo(f"[deforest] wrote {len(merged['features'])} polygons → {out_path}")


@main.command()
def runtime():
    """Print detected hardware and the autoscaled training/inference defaults."""
    from .runtime import autoscale_defaults, detect_hardware

    hw = detect_hardware()
    click.echo(hw.summary())
    scales = autoscale_defaults(hw)
    click.echo("\nautoscaled defaults:")
    for field in [
        "torch_device", "amp_dtype", "batch_size", "num_workers",
        "prefetch_factor", "pin_memory", "persistent_workers",
        "lgbm_threads", "preprocess_workers", "max_cached_patches",
    ]:
        click.echo(f"  {field}: {getattr(scales, field)}")


@main.command("preprocess")
@click.option("--config", type=click.Path(path_type=Path, exists=True), default=Path("configs/server.yaml"))
@click.option("--split", type=click.Choice(["train", "test"]), default="train")
@click.option("--tiles", type=str, default=None)
@click.option("--workers", type=int, default=None)
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None)
def preprocess_cmd(config: Path, split: str, tiles: str | None, workers: int | None, cache_dir: Path | None):
    """Pre-compute per-tile feature + label caches under /mnt/scratch."""
    import subprocess, sys
    args = [sys.executable, "scripts/preprocess_tiles.py", "--config", str(config), "--split", split]
    if tiles:
        args += ["--tiles", tiles]
    if workers is not None:
        args += ["--workers", str(workers)]
    if cache_dir is not None:
        args += ["--cache-dir", str(cache_dir)]
    sys.exit(subprocess.call(args))


@main.command("train-deep")
@click.option("--config", type=click.Path(path_type=Path, exists=True), default=Path("configs/server.yaml"))
@click.option("--val-tiles", type=str, default=None)
@click.option("--resume", type=click.Path(path_type=Path), default=None)
def train_deep_cmd(config: Path, val_tiles: str | None, resume: Path | None):
    """Train ChangeUNet on the precomputed patch cache."""
    import subprocess, sys
    args = [sys.executable, "scripts/train_deep.py", "--config", str(config)]
    if val_tiles:
        args += ["--val-tiles", val_tiles]
    if resume is not None:
        args += ["--resume", str(resume)]
    sys.exit(subprocess.call(args))


@main.command("predict-ensemble")
@click.option("--config", type=click.Path(path_type=Path, exists=True), default=Path("configs/server.yaml"))
@click.option("--deep-ckpt", type=click.Path(path_type=Path), default=None)
@click.option("--gbm-model", type=click.Path(path_type=Path), default=None)
@click.option("--split", type=click.Choice(["train", "test"]), default="test")
@click.option("--tiles", type=str, default=None)
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
def predict_ensemble_cmd(
    config: Path,
    deep_ckpt: Path | None,
    gbm_model: Path | None,
    split: str,
    tiles: str | None,
    out_path: Path,
):
    """Ensemble deep + GBM predictions into a final submission."""
    import subprocess, sys
    args = [sys.executable, "scripts/predict_ensemble.py",
            "--config", str(config), "--split", split, "--out", str(out_path)]
    if deep_ckpt is not None:
        args += ["--deep-ckpt", str(deep_ckpt)]
    if gbm_model is not None:
        args += ["--gbm-model", str(gbm_model)]
    if tiles:
        args += ["--tiles", tiles]
    sys.exit(subprocess.call(args))


@main.command()
@click.option("--predictions", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--ground-truth", type=click.Path(path_type=Path, exists=True), required=True)
def evaluate(predictions: Path, ground_truth: Path):
    """Compute Union IoU + polygon metrics against a GT GeoJSON."""
    from .evaluation.metrics import evaluate as _eval

    pred_fc = json.loads(predictions.read_text())
    gt_fc = json.loads(ground_truth.read_text())
    res = _eval(pred_fc, gt_fc)
    click.echo(json.dumps(res.as_dict(), indent=2))


# --- helpers ---------------------------------------------------------------


def _data_paths(cfg) -> DataPaths:
    d = cfg.raw["data"]
    return DataPaths(
        root=Path(d["root"]),
        s1_subdir=d["s1_subdir"],
        s2_subdir=d["s2_subdir"],
        aef_subdir=d["aef_subdir"],
        labels_subdir=d["labels_subdir"],
    )


def _resolve_tiles(tiles: str | None, cfg, split: str) -> list[str]:
    if tiles:
        return [t.strip() for t in tiles.split(",") if t.strip()]
    data_root = Path(cfg.raw["data"]["root"])
    # Prefer the metadata GeoJSON if it exists
    key = "train_tiles_geojson" if split == "train" else "test_tiles_geojson"
    meta_path = Path(cfg.raw["data"].get(key, ""))
    if meta_path.exists():
        ids = list_tiles(meta_path)
        if ids:
            return ids
    return discover_tiles(data_root, split=split)


if __name__ == "__main__":
    main()
