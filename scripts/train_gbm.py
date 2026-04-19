"""Train the improved LightGBM pixel classifier.

Two-pass design so global stratified quotas (Improvement 1) can be allocated
correctly before we commit features into memory:

1. **Pass A** iterates over every training tile, builds features + fused
   labels, runs :func:`select_candidate_pixels` and keeps only the
   *indices* of candidate positives / negatives per tile (plus each
   positive's confidence and agreement count).
2. Given the per-tile / per-region candidate pools, allocate

   - ``total_pos`` positives across regions, proportional to
     ``sqrt(region_pos_count)``,
   - ``total_neg`` negatives across regions analogously,

   and within each tile stratify positives across confidence quantiles.
3. **Pass B** rebuilds the features per tile *once more* and slices them at
   the chosen indices. (Features aren't cached in RAM between passes to
   keep memory usage bounded.)
4. Fit the LightGBM with `objective=cross_entropy`, passing the validation
   split so early stopping kicks in.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running ``python scripts/train_gbm.py`` without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deforest2.config import load_config  # noqa: E402
from deforest2.data.paths import DataPaths, discover_tiles, list_tiles, region_of  # noqa: E402
from deforest2.features.aef import aef_features  # noqa: E402
from deforest2.features.satellite import (  # noqa: E402
    pack_features,
    s1_annual_stats,
    s2_annual_stats,
    s2_worst_drop_across_years,
)
from deforest2.features.terrain import terrain_features  # noqa: E402
from deforest2.inference.tile_predict import (  # noqa: E402
    _build_forest_mask_2020,
    _load_and_fuse_labels,
)
from deforest2.models.gbm import (  # noqa: E402
    GBMConfig,
    PixelGBM,
    compute_positive_weights,
    select_candidate_pixels,
    subsample_positives_stratified,
)

import rasterio  # noqa: E402
from rasterio.warp import Resampling  # noqa: E402

from deforest2.data.align import reproject_multiband_to_grid  # noqa: E402
from deforest2.data.readers import list_s2_months, read_aef  # noqa: E402


# ---------------------------------------------------------------------------
# Tile selection (held-out validation by region)
# ---------------------------------------------------------------------------


def _split_train_val(
    tile_ids: list[str],
    val_regions: list[str],
    *,
    prefix_chars: int = 5,
) -> tuple[list[str], list[str]]:
    val: list[str] = []
    trn: list[str] = []
    regex = [re.compile(p) for p in val_regions or []]
    for tid in tile_ids:
        region = region_of(tid, prefix_chars=prefix_chars)
        if any(r.fullmatch(region) or r.search(tid) for r in regex):
            val.append(tid)
        else:
            trn.append(tid)
    return trn, val


# ---------------------------------------------------------------------------
# Per-tile feature + label build
# ---------------------------------------------------------------------------


def _build_tile_inputs(
    tid: str,
    paths: DataPaths,
    fusion_cfg: dict,
    feature_cfg: dict,
) -> dict | None:
    """Build features, fused labels and the 2020 forest mask for one tile.

    Returns ``None`` on skippable tiles (missing S2 or AEF).
    """
    s2_dir = paths.s2_dir(tid, split="train")
    s2_entries = list_s2_months(s2_dir)
    if not s2_entries:
        return None
    ref_tif = paths.s2_tif(tid, s2_entries[0][0], s2_entries[0][1], split="train")
    with rasterio.open(ref_tif) as src:
        ref_crs, ref_transform, ref_shape = src.crs, src.transform, (src.height, src.width)

    years_available = sorted({y for (y, _) in s2_entries})
    year_base = years_available[0]
    year_last = years_available[-1]

    aef_by_year: dict[int, np.ndarray] = {}
    for y in years_available:
        p = paths.aef_tiff(tid, y, split="train")
        if not p.exists():
            continue
        data, profile = read_aef(p)
        aef_by_year[y] = reproject_multiband_to_grid(
            data,
            src_transform=profile["transform"], src_crs=profile["crs"],
            dst_transform=ref_transform, dst_crs=ref_crs,
            dst_shape=ref_shape, resampling=Resampling.bilinear,
        )
    if not aef_by_year:
        return None

    aef_feats = aef_features(
        aef_by_year, multi_year_drift=bool(feature_cfg.get("aef_multi_year_drift", True))
    )

    grid = dict(ref_transform=ref_transform, ref_crs=ref_crs, ref_shape=ref_shape)
    pct = feature_cfg.get("s2_percentiles", [10, 90])
    percentiles = tuple(pct) if pct else None
    include_std = bool(feature_cfg.get("s2_intra_year_std", True))

    s1_base = s1_annual_stats(paths.s1_dir(tid, split="train"), year_base, **grid)
    s1_last = s1_annual_stats(paths.s1_dir(tid, split="train"), year_last, **grid)
    s2_base = s2_annual_stats(
        s2_dir, year_base, percentiles=percentiles, include_intra_year_std=include_std, **grid
    )
    s2_last = s2_annual_stats(
        s2_dir, year_last, percentiles=percentiles, include_intra_year_std=include_std, **grid
    )

    s2_multi = None
    if feature_cfg.get("s2_worst_drop", True):
        s2_multi = s2_worst_drop_across_years(
            s2_dir,
            year_base=year_base,
            years=years_available,
            ref_transform=ref_transform, ref_crs=ref_crs, ref_shape=ref_shape,
        )

    terr = None
    if feature_cfg.get("use_terrain", False):
        terr = terrain_features(
            ref_transform, ref_crs, ref_shape,
            cache_dir=str(feature_cfg.get("terrain_cache_dir", "cache/dem")),
        )

    X, names = pack_features(
        aef_feats, s2_base, s2_last, s1_base, s1_last,
        s2_multi=s2_multi, terrain=terr,
    )

    fused = _load_and_fuse_labels(
        tid, paths,
        ref_crs=ref_crs, ref_transform=ref_transform, ref_shape=ref_shape,
        fusion_cfg=fusion_cfg,
    )
    if fused is None:
        return None
    forest = _build_forest_mask_2020(aef_by_year, year_base, s2_base)

    return {
        "tile_id": tid,
        "X": X,
        "names": names,
        "fused": fused,
        "forest": forest,
    }


# ---------------------------------------------------------------------------
# Quota allocation 
# ---------------------------------------------------------------------------


def _sqrt_region_allocation(region_counts: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across regions proportionally to ``sqrt(count)``."""
    weights = {r: math.sqrt(max(0, n)) for r, n in region_counts.items()}
    s = sum(weights.values())
    if s <= 0:
        return {r: 0 for r in region_counts}
    return {r: int(round(total * (w / s))) for r, w in weights.items()}


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out", type=Path, default=Path("models/gbm.txt"))
    ap.add_argument(
        "--tiles", type=str, default=None,
        help="Comma-separated tile ids to restrict training (default: all train tiles).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.raw["data"]
    gcfg = cfg.raw["gbm"]
    feature_cfg = cfg.raw.get("features", {}) or {}
    fusion_cfg = {
        "agreement_threshold": cfg.raw["label_fusion"]["agreement_threshold"],
        "single_threshold": cfg.raw["label_fusion"]["single_threshold"],
    }
    paths = DataPaths(
        root=Path(data_cfg["root"]),
        s1_subdir=data_cfg.get("s1_subdir", "sentinel-1"),
        s2_subdir=data_cfg.get("s2_subdir", "sentinel-2"),
        aef_subdir=data_cfg.get("aef_subdir", "aef-embeddings"),
        labels_subdir=data_cfg.get("labels_subdir", "labels/train"),
    )

    if args.tiles:
        tile_ids = [t.strip() for t in args.tiles.split(",") if t.strip()]
    else:
        meta_path = Path(data_cfg.get("train_tiles_geojson", ""))
        tile_ids = list_tiles(meta_path) if meta_path.exists() else []
        if not tile_ids:
            tile_ids = discover_tiles(Path(data_cfg["root"]), split="train")
    if not tile_ids:
        raise SystemExit("No training tiles found.")

    prefix_chars = int(gcfg.get("region_prefix_chars", 5))
    train_ids, val_ids = _split_train_val(
        tile_ids, gcfg.get("val_regions", []) or [], prefix_chars=prefix_chars
    )
    print(f"[train] {len(train_ids)} train tiles, {len(val_ids)} val tiles")
    if not train_ids:
        raise SystemExit("Validation split left zero training tiles.")

    hard_neg_max = float(cfg.raw["label_fusion"].get("hard_negative_max", 0.2))
    rng = np.random.default_rng(int(gcfg.get("seed", 1337)))

    # ---------------- Pass A: candidate collection ------------------------
    print("[train] pass A: scanning training tiles for candidate pixels…")
    pre: dict[str, dict] = {}
    feature_names: list[str] | None = None
    for tid in tqdm(train_ids):
        built = _build_tile_inputs(tid, paths, fusion_cfg, feature_cfg)
        if built is None:
            continue
        feature_names = built["names"]
        fused = built["fused"]
        forest = built["forest"]
        pos_idx, neg_idx, pos_conf, pos_agree, pos_days = select_candidate_pixels(
            fused.binary,
            fused.max_confidence,
            fused.agree_count,
            fused.median_days,
            forest,
            hard_negative_max=hard_neg_max,
        )
        pre[tid] = {
            "region": region_of(tid, prefix_chars=prefix_chars),
            "pos_idx": pos_idx,
            "neg_idx": neg_idx,
            "pos_conf": pos_conf,
            "pos_agree": pos_agree,
            "pos_days": pos_days,
            "n_pixels": int(fused.binary.size),
        }

    if not pre:
        raise SystemExit("No usable training tiles (missing S2 / AEF / labels).")

    total_pos = int(gcfg.get("total_pos", 200_000))
    total_neg = int(gcfg.get("total_neg", 400_000))
    n_strata = int(gcfg.get("confidence_strata", 4))

    # Region-level pools
    region_pos: dict[str, int] = {}
    region_neg: dict[str, int] = {}
    for tid, info in pre.items():
        region_pos[info["region"]] = region_pos.get(info["region"], 0) + int(info["pos_idx"].size)
        region_neg[info["region"]] = region_neg.get(info["region"], 0) + int(info["neg_idx"].size)
    alloc_pos = _sqrt_region_allocation(region_pos, total_pos)
    alloc_neg = _sqrt_region_allocation(region_neg, total_neg)

    # Tile-level caps inside each region proportional to counts.
    caps: dict[str, tuple[int, int]] = {}
    for region, a_pos in alloc_pos.items():
        tiles = [tid for tid, info in pre.items() if info["region"] == region]
        reg_total_pos = sum(int(pre[t]["pos_idx"].size) for t in tiles)
        reg_total_neg = sum(int(pre[t]["neg_idx"].size) for t in tiles)
        for t in tiles:
            p = int(pre[t]["pos_idx"].size)
            n = int(pre[t]["neg_idx"].size)
            cap_p = int(round(a_pos * (p / reg_total_pos))) if reg_total_pos > 0 else 0
            cap_n = int(round(alloc_neg.get(region, 0) * (n / reg_total_neg))) if reg_total_neg > 0 else 0
            caps[t] = (cap_p, cap_n)

    # ---------------- Pass B: rebuild features and slice -------------------
    print("[train] pass B: slicing features at chosen indices…")
    use_soft = str(gcfg.get("objective", "cross_entropy")) == "cross_entropy"
    X_list, y_list, w_list = [], [], []
    per_tile_region: list[str] = []
    per_tile_counts: dict[str, int] = {}
    for tid in tqdm(list(pre.keys())):
        cap_p, cap_n = caps[tid]
        if cap_p <= 0 and cap_n <= 0:
            continue
        built = _build_tile_inputs(tid, paths, fusion_cfg, feature_cfg)
        if built is None:
            continue
        X = built["X"]
        info = pre[tid]
        pos_idx = info["pos_idx"]
        neg_idx = info["neg_idx"]
        pos_conf = info["pos_conf"]
        pos_agree = info["pos_agree"]

        if cap_p > 0 and pos_idx.size > 0:
            keep_pos_pos = subsample_positives_stratified(
                pos_conf, cap_p, n_strata=n_strata, rng=rng
            )
            keep_pos = pos_idx[keep_pos_pos]
            keep_pos_conf = pos_conf[keep_pos_pos]
            keep_pos_agree = pos_agree[keep_pos_pos]
        else:
            keep_pos = np.empty(0, dtype=pos_idx.dtype)
            keep_pos_conf = np.empty(0, dtype=np.float32)
            keep_pos_agree = np.empty(0, dtype=np.uint8)
        if cap_n > 0 and neg_idx.size > 0:
            keep_neg = rng.choice(neg_idx, size=min(cap_n, neg_idx.size), replace=False)
        else:
            keep_neg = np.empty(0, dtype=neg_idx.dtype)
        idx = np.concatenate([keep_pos, keep_neg])
        if idx.size == 0:
            continue

        Xs = X[idx].astype(np.float32, copy=False)
        # Positive targets: soft confidence for cross_entropy, hard 1 for binary.
        ys = np.zeros(idx.shape[0], dtype=np.float32)
        if use_soft:
            ys[: keep_pos.size] = keep_pos_conf
        else:
            ys[: keep_pos.size] = 1.0

        w_pos = compute_positive_weights(keep_pos_conf, keep_pos_agree)
        w_neg = np.ones(keep_neg.size, dtype=np.float32)
        ws = np.concatenate([w_pos, w_neg]).astype(np.float32)

        if bool(gcfg.get("normalise_tile_weights", True)) and ws.size:
            m = float(ws.mean())
            if m > 1e-6:
                ws = ws / m

        X_list.append(Xs)
        y_list.append(ys)
        w_list.append(ws)
        per_tile_region.append(pre[tid]["region"])
        per_tile_counts[pre[tid]["region"]] = per_tile_counts.get(pre[tid]["region"], 0) + ws.size

    if not X_list:
        raise SystemExit("No pixels selected after quota allocation.")

    # multiply in region-balance weight.
    if bool(gcfg.get("region_balance", True)) and per_tile_counts:
        n_regions = len(per_tile_counts)
        total_samples = sum(per_tile_counts.values())
        for i, region in enumerate(per_tile_region):
            rw = total_samples / max(1, n_regions * per_tile_counts[region])
            w_list[i] *= float(rw)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    w = np.concatenate(w_list, axis=0)
    print(
        f"[train] dataset: X={X.shape}, pos={(y > 0).sum()}, neg={(y == 0).sum()}, "
        f"feat_count={len(feature_names or [])}"
    )

    # ---------------- Validation set --------------------------------------
    X_val = y_val = w_val = None
    if val_ids:
        print("[train] building validation features…")
        Xv_list, yv_list, wv_list = [], [], []
        for tid in tqdm(val_ids):
            built = _build_tile_inputs(tid, paths, fusion_cfg, feature_cfg)
            if built is None:
                continue
            fused = built["fused"]
            forest = built["forest"]
            pos_idx, neg_idx, pos_conf, pos_agree, _ = select_candidate_pixels(
                fused.binary,
                fused.max_confidence,
                fused.agree_count,
                fused.median_days,
                forest,
                hard_negative_max=hard_neg_max,
            )
            # Cap per-tile contributions to keep memory predictable.
            cap_p = min(pos_idx.size, 20_000)
            cap_n = min(neg_idx.size, 40_000)
            rng_v = np.random.default_rng(0)
            keep_pos = rng_v.choice(pos_idx, size=cap_p, replace=False) if cap_p else np.empty(0, dtype=int)
            keep_neg = rng_v.choice(neg_idx, size=cap_n, replace=False) if cap_n else np.empty(0, dtype=int)
            idx = np.concatenate([keep_pos, keep_neg])
            if idx.size == 0:
                continue
            Xs = built["X"][idx].astype(np.float32, copy=False)
            ys = np.zeros(idx.shape[0], dtype=np.float32)
            ys[: keep_pos.size] = 1.0
            ws = np.ones(idx.shape[0], dtype=np.float32)
            Xv_list.append(Xs)
            yv_list.append(ys)
            wv_list.append(ws)
        if Xv_list:
            X_val = np.concatenate(Xv_list, axis=0)
            y_val = np.concatenate(yv_list, axis=0)
            w_val = np.concatenate(wv_list, axis=0)
            print(f"[train] val dataset: X={X_val.shape}, pos={(y_val > 0).sum()}")

    # ---------------- Fit -------------------------------------------------
    cfg_gbm = GBMConfig(
        objective=str(gcfg.get("objective", "cross_entropy")),
        n_estimators=int(gcfg.get("n_estimators", 2000)),
        early_stopping_rounds=int(gcfg.get("early_stopping_rounds", 50)),
        learning_rate=float(gcfg.get("learning_rate", 0.05)),
        num_leaves=int(gcfg.get("num_leaves", 63)),
        min_child_samples=int(gcfg.get("min_child_samples", 200)),
        feature_fraction=float(gcfg.get("feature_fraction", 0.9)),
        bagging_fraction=float(gcfg.get("bagging_fraction", 0.9)),
        bagging_freq=int(gcfg.get("bagging_freq", 5)),
        seed=int(gcfg.get("seed", 1337)),
    )
    if cfg_gbm.objective == "binary" and bool(gcfg.get("use_scale_pos_weight", True)):
        pos_frac = float((y > 0).mean())
        if 0 < pos_frac < 1:
            cfg_gbm.scale_pos_weight = (1.0 - pos_frac) / pos_frac

    print(
        f"[train] fitting LightGBM ({cfg_gbm.objective}, lr={cfg_gbm.learning_rate}, "
        f"n_estimators<={cfg_gbm.n_estimators}, early_stopping={cfg_gbm.early_stopping_rounds})"
    )
    model = PixelGBM(cfg_gbm).fit(
        X, y, weights=w, feature_names=feature_names,
        eval_X=X_val, eval_y=y_val, eval_w=w_val,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)

    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta = {
        "feature_names": feature_names,
        "val_regions": gcfg.get("val_regions", []),
        "train_tiles": list(pre.keys()),
        "val_tiles": val_ids,
        "best_iteration": model.best_iteration_,
        "objective": cfg_gbm.objective,
        "n_pos": int((y > 0).sum()),
        "n_neg": int((y == 0).sum()),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] wrote {args.out} (meta → {meta_path.name})")


if __name__ == "__main__":
    main()
