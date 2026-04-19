"""Visualize a submission GeoJSON over the test-tile Sentinel-2 basemap.

For every test tile (or a user-supplied subset) the script

1. resolves the Sentinel-2 L2A composites available for that tile,
2. builds an RGB basemap (B04/B03/B02) from a chosen month (or the
   per-pixel annual median across months) and stretches it for display,
3. reprojects the submission polygons from EPSG:4326 into the tile CRS
   and clips them to the tile footprint, and
4. renders the tile + overlaid submission polygons as a PNG.

An overview figure with all tile footprints + submission polygons on a
simple EPSG:4326 axis is also produced for a quick global sanity check.

Example::

    python scripts/visualize_submission.py \
        --config configs/default.yaml \
        --submission submissions/submission.geojson \
        --out-dir submissions/viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from shapely.geometry import box, shape  # noqa: E402
from shapely.ops import transform as shp_transform  # noqa: E402
from tqdm import tqdm  # noqa: E402

import geopandas as gpd  # noqa: E402

from deforest2.config import load_config  # noqa: E402
from deforest2.data.paths import DataPaths, list_tiles  # noqa: E402
from deforest2.data.readers import list_s2_months  # noqa: E402


# Sentinel-2 L2A band order in the challenge dataset:
#   B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
# so RGB = (B04, B03, B02) at indices (3, 2, 1).
_RGB_IDX = (3, 2, 1)
_S2_SCALE = 10_000.0


# ---------------------------------------------------------------------------
# Submission loading / spatial indexing
# ---------------------------------------------------------------------------


def _load_submission(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


# ---------------------------------------------------------------------------
# Sentinel-2 RGB basemap
# ---------------------------------------------------------------------------


def _pick_s2_tif(
    paths: DataPaths,
    tile_id: str,
    split: str,
    *,
    year: int | None,
    month: int | None,
) -> Path | None:
    """Pick a Sentinel-2 GeoTIFF to render as the basemap.

    Preference order: exact ``(year, month)`` if given, else the latest
    cloud-freeish month (highest ``month`` of the latest ``year``).
    """
    months = list_s2_months(paths.s2_dir(tile_id, split=split))
    if not months:
        return None
    if year is not None and month is not None:
        for y, m in months:
            if y == year and m == month:
                return paths.s2_tif(tile_id, y, m, split=split)
    if year is not None:
        candidates = [(y, m) for (y, m) in months if y == year]
        if candidates:
            y, m = max(candidates)
            return paths.s2_tif(tile_id, y, m, split=split)
    y, m = max(months)
    return paths.s2_tif(tile_id, y, m, split=split)


def _s2_median_rgb(
    paths: DataPaths,
    tile_id: str,
    split: str,
    *,
    year: int,
) -> tuple[np.ndarray, dict] | None:
    """Per-pixel median RGB across all months of ``year`` (ignoring 0 nodata)."""
    months = [m for (y, m) in list_s2_months(paths.s2_dir(tile_id, split=split)) if y == year]
    if not months:
        return None
    stacks: list[np.ndarray] = []
    profile: dict | None = None
    for m in sorted(months):
        tif = paths.s2_tif(tile_id, year, m, split=split)
        if not tif.exists():
            continue
        with rasterio.open(tif) as src:
            rgb = src.read([i + 1 for i in _RGB_IDX]).astype(np.float32)
            if profile is None:
                profile = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "shape": (src.height, src.width),
                }
        rgb = rgb / _S2_SCALE
        rgb[rgb <= 0] = np.nan
        stacks.append(rgb)
    if not stacks or profile is None:
        return None
    arr = np.stack(stacks, axis=0)  # (T, 3, H, W)
    med = np.nanmedian(arr, axis=0)
    return med, profile


def _load_rgb_basemap(
    paths: DataPaths,
    tile_id: str,
    split: str,
    *,
    year: int | None,
    month: int | None,
    mode: str,
) -> tuple[np.ndarray, dict] | None:
    """Return ``(rgb_float32 ∈ [0,1], profile)`` or ``None`` if no S2 data."""
    if mode == "median":
        # Pick a year: requested, else the last available.
        months = list_s2_months(paths.s2_dir(tile_id, split=split))
        if not months:
            return None
        chosen_year = year if year is not None else max(y for (y, _) in months)
        pair = _s2_median_rgb(paths, tile_id, split, year=chosen_year)
        if pair is not None:
            rgb, prof = pair
        else:
            return None
    else:
        tif = _pick_s2_tif(paths, tile_id, split, year=year, month=month)
        if tif is None or not tif.exists():
            return None
        with rasterio.open(tif) as src:
            rgb = src.read([i + 1 for i in _RGB_IDX]).astype(np.float32) / _S2_SCALE
            prof = {
                "transform": src.transform,
                "crs": src.crs,
                "shape": (src.height, src.width),
            }
        rgb[rgb <= 0] = np.nan

    stretched = _stretch_rgb(rgb)
    return stretched, prof


def _stretch_rgb(rgb: np.ndarray, *, lo_q: float = 2.0, hi_q: float = 98.0) -> np.ndarray:
    """Per-channel percentile stretch → [0, 1] with NaN→0 for display."""
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[0]):
        band = rgb[c]
        valid = np.isfinite(band)
        if not valid.any():
            continue
        lo = np.nanpercentile(band, lo_q)
        hi = np.nanpercentile(band, hi_q)
        if hi <= lo:
            continue
        band = (band - lo) / (hi - lo)
        out[c] = np.clip(np.nan_to_num(band, nan=0.0), 0.0, 1.0)
    # (3, H, W) → (H, W, 3) for imshow
    return np.transpose(out, (1, 2, 0))


# ---------------------------------------------------------------------------
# Per-tile plotting
# ---------------------------------------------------------------------------


def _clip_submission_to_tile(
    submission_4326: gpd.GeoDataFrame,
    tile_profile: dict,
) -> gpd.GeoDataFrame:
    """Reproject the submission to tile CRS and keep polygons touching the tile."""
    tile_crs = tile_profile["crs"]
    transform = tile_profile["transform"]
    h, w = tile_profile["shape"]
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    tile_box = box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))
    if submission_4326.empty:
        return submission_4326.copy()
    sub_utm = submission_4326.to_crs(tile_crs)
    mask = sub_utm.intersects(tile_box)
    return sub_utm[mask].copy()


def _plot_tile(
    tile_id: str,
    rgb: np.ndarray,
    profile: dict,
    submission_clip: gpd.GeoDataFrame,
    out_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    transform = profile["transform"]
    h, w = profile["shape"]
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    extent = (min(left, right), max(left, right), min(top, bottom), max(top, bottom))

    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="nearest")

    # Tile footprint for context.
    ax.add_patch(
        Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            fill=False,
            edgecolor="white",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
        )
    )

    if not submission_clip.empty:
        submission_clip.plot(
            ax=ax,
            facecolor=(1.0, 0.2, 0.2, 0.25),
            edgecolor="yellow",
            linewidth=1.2,
        )
        if "time_step" in submission_clip.columns:
            for _, row in submission_clip.iterrows():
                ts = row.get("time_step")
                if ts in (None, ""):
                    continue
                c = row.geometry.centroid
                ax.annotate(
                    str(ts),
                    xy=(c.x, c.y),
                    color="yellow",
                    fontsize=7,
                    ha="center",
                    va="center",
                )

    n = 0 if submission_clip.empty else len(submission_clip)
    title = f"{tile_id}  —  {n} submission polygon(s)"
    if title_suffix:
        title += f"  ({title_suffix})"
    ax.set_title(title)
    ax.set_xlabel(f"easting  [{profile['crs']}]")
    ax.set_ylabel("northing")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Overview figure across all tiles
# ---------------------------------------------------------------------------


def _tile_bounds_4326(paths: DataPaths, tile_id: str, split: str) -> tuple[float, float, float, float] | None:
    """Fast footprint: read any one S2 tif's bounds and reproject to 4326."""
    months = list_s2_months(paths.s2_dir(tile_id, split=split))
    if not months:
        return None
    y, m = months[0]
    tif = paths.s2_tif(tile_id, y, m, split=split)
    if not tif.exists():
        return None
    with rasterio.open(tif) as src:
        return tuple(transform_bounds(src.crs, "EPSG:4326", *src.bounds))  # type: ignore[return-value]


def _plot_overview(
    tile_ids: list[str],
    paths: DataPaths,
    split: str,
    submission_4326: gpd.GeoDataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    any_drawn = False
    for tid in tile_ids:
        b = _tile_bounds_4326(paths, tid, split)
        if b is None:
            continue
        minx, miny, maxx, maxy = b
        ax.add_patch(
            Rectangle(
                (minx, miny),
                maxx - minx,
                maxy - miny,
                fill=False,
                edgecolor="steelblue",
                linewidth=1.2,
            )
        )
        ax.annotate(tid, xy=((minx + maxx) / 2, maxy), fontsize=7, ha="center", va="bottom", color="steelblue")
        any_drawn = True

    if not submission_4326.empty:
        submission_4326.plot(
            ax=ax,
            facecolor=(1.0, 0.2, 0.2, 0.6),
            edgecolor="red",
            linewidth=0.4,
        )

    n = 0 if submission_4326.empty else len(submission_4326)
    ax.set_title(f"Submission overview — {len(tile_ids)} tiles, {n} polygons")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if any_drawn:
        ax.autoscale_view()
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_tiles(tiles_arg: str | None, cfg, split: str, paths: DataPaths) -> list[str]:
    if tiles_arg:
        return [t.strip() for t in tiles_arg.split(",") if t.strip()]
    data_cfg = cfg.raw["data"]
    key = "train_tiles_geojson" if split == "train" else "test_tiles_geojson"
    meta_path = Path(data_cfg.get(key, ""))
    if meta_path.exists():
        ids = list_tiles(meta_path)
        if ids:
            return ids
    # Fallback: look at what exists on disk.
    s2_root = paths.root / paths.s2_subdir / split
    if not s2_root.exists():
        return []
    return sorted(
        p.name[: -len("__s2_l2a")]
        for p in s2_root.iterdir()
        if p.is_dir() and p.name.endswith("__s2_l2a")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument(
        "--submission",
        type=Path,
        default=Path("submissions/submission.geojson"),
        help="Submission GeoJSON to overlay on the test data.",
    )
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--tiles", type=str, default=None, help="Comma-separated tile ids (defaults to all).")
    ap.add_argument("--out-dir", type=Path, default=Path("submissions/viz"))
    ap.add_argument(
        "--basemap",
        choices=["single", "median"],
        default="median",
        help="'single' uses one S2 GeoTIFF; 'median' builds a per-pixel median across all months of --year.",
    )
    ap.add_argument("--year", type=int, default=None, help="S2 year for the basemap (default: latest available).")
    ap.add_argument("--month", type=int, default=None, help="S2 month (used when --basemap single).")
    ap.add_argument("--no-overview", action="store_true", help="Skip the global overview figure.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.raw["data"]
    paths = DataPaths(
        root=Path(data_cfg["root"]),
        s1_subdir=data_cfg.get("s1_subdir", "sentinel-1"),
        s2_subdir=data_cfg.get("s2_subdir", "sentinel-2"),
        aef_subdir=data_cfg.get("aef_subdir", "aef-embeddings"),
        labels_subdir=data_cfg.get("labels_subdir", "labels/train"),
    )

    if not args.submission.exists():
        raise SystemExit(f"Submission not found: {args.submission}")
    submission_4326 = _load_submission(args.submission)
    print(
        f"[viz] loaded submission: {len(submission_4326)} polygons from {args.submission}"
    )

    tile_ids = _resolve_tiles(args.tiles, cfg, args.split, paths)
    if not tile_ids:
        raise SystemExit(f"No tiles found for split={args.split}")
    print(f"[viz] visualising {len(tile_ids)} tile(s) → {args.out_dir}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    for tid in tqdm(tile_ids):
        basemap = _load_rgb_basemap(
            paths, tid, args.split,
            year=args.year,
            month=args.month,
            mode=args.basemap,
        )
        if basemap is None:
            skipped.append(tid)
            continue
        rgb, profile = basemap
        clip = _clip_submission_to_tile(submission_4326, profile)
        suffix_bits: list[str] = [args.basemap]
        if args.year is not None:
            suffix_bits.append(str(args.year))
        if args.basemap == "single" and args.month is not None:
            suffix_bits.append(f"m{args.month}")
        _plot_tile(
            tid, rgb, profile, clip,
            out_dir / f"{tid}.png",
            title_suffix=" ".join(suffix_bits),
        )

    if skipped:
        print(f"[viz] skipped {len(skipped)} tile(s) with no Sentinel-2 basemap: {skipped}")

    if not args.no_overview:
        _plot_overview(tile_ids, paths, args.split, submission_4326, out_dir / "_overview.png")
        print(f"[viz] wrote overview → {out_dir / '_overview.png'}")

    print(f"[viz] done → {out_dir}")


if __name__ == "__main__":
    main()
