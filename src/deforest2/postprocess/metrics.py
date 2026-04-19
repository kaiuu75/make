"""Lightweight evaluation helpers used by the tuning + CV scripts.

We report three quantities that approximate the leaderboard metric
(polygon Union IoU) and the two sanity metrics:

* ``union_iou(pred, gt)`` — classical IoU of the union of predicted polygons
  against the union of ground-truth polygons.
* ``polygon_recall`` — fraction of GT polygons that intersect any prediction.
* ``polygon_fpr_proxy`` — fraction of predicted polygons that do not
  intersect any GT polygon (a cheap proxy for polygon-level false positives).

All operations are performed in the dataset's native UTM CRS (via
``estimate_utm_crs``) so areas are metric-accurate.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union


@dataclass
class EvalResult:
    union_iou: float
    polygon_recall: float
    polygon_fpr_proxy: float
    n_pred: int
    n_gt: int

    def as_dict(self) -> dict:
        return {
            "union_iou": self.union_iou,
            "polygon_recall": self.polygon_recall,
            "polygon_fpr_proxy": self.polygon_fpr_proxy,
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
        }


def evaluate(pred_fc: dict, gt_fc: dict) -> EvalResult:
    pred_geoms = [shape(f["geometry"]) for f in pred_fc.get("features", []) if f.get("geometry")]
    gt_geoms = [shape(f["geometry"]) for f in gt_fc.get("features", []) if f.get("geometry")]

    n_pred = len(pred_geoms)
    n_gt = len(gt_geoms)

    if n_pred == 0 and n_gt == 0:
        return EvalResult(1.0, 1.0, 0.0, 0, 0)

    # Project to a metric CRS before area math. 4326 → UTM via GeoPandas.
    gdf_p = gpd.GeoDataFrame(geometry=pred_geoms, crs="EPSG:4326") if pred_geoms else None
    gdf_g = gpd.GeoDataFrame(geometry=gt_geoms, crs="EPSG:4326") if gt_geoms else None
    if gdf_p is not None:
        utm = gdf_p.estimate_utm_crs()
    elif gdf_g is not None:
        utm = gdf_g.estimate_utm_crs()
    else:
        return EvalResult(1.0, 1.0, 0.0, 0, 0)

    pred_u = gdf_p.to_crs(utm).geometry if gdf_p is not None else None
    gt_u = gdf_g.to_crs(utm).geometry if gdf_g is not None else None

    pred_union = unary_union(list(pred_u)) if pred_u is not None and n_pred else None
    gt_union = unary_union(list(gt_u)) if gt_u is not None and n_gt else None

    if pred_union is None and gt_union is None:
        union_iou = 1.0
    elif pred_union is None or gt_union is None:
        union_iou = 0.0
    else:
        inter = pred_union.intersection(gt_union).area
        union = pred_union.union(gt_union).area
        union_iou = float(inter / union) if union > 0 else 0.0

    # Polygon-level recall and FPR proxy use metric-CRS geometries.
    if n_gt and pred_u is not None and n_pred:
        hit = sum(1 for g in gt_u if any(g.intersects(p) for p in pred_u))
        recall = hit / n_gt
    elif n_gt:
        recall = 0.0
    else:
        recall = 1.0

    if n_pred and gt_u is not None and n_gt:
        miss = sum(1 for p in pred_u if not any(p.intersects(g) for g in gt_u))
        fpr = miss / n_pred
    elif n_pred:
        fpr = 1.0
    else:
        fpr = 0.0

    return EvalResult(
        union_iou=float(union_iou),
        polygon_recall=float(recall),
        polygon_fpr_proxy=float(fpr),
        n_pred=n_pred,
        n_gt=n_gt,
    )


def fused_labels_to_feature_collection(
    binary: "gpd.array",
    *,
    transform,
    crs,
    min_area_ha: float = 0.5,
) -> dict:
    """Turn a fused binary label raster into an EPSG:4326 FeatureCollection.

    Used by the tuner / CV harness to create a pseudo-ground-truth from the
    weak labels of the held-out tiles. It's not real ground truth, but it's
    the best proxy we have without a labelled hold-out.
    """
    import json
    import numpy as np
    from rasterio.features import shapes

    bin_u8 = (binary > 0).astype(np.uint8)
    if bin_u8.sum() == 0:
        return {"type": "FeatureCollection", "features": []}

    feats = []
    for geom, value in shapes(bin_u8, mask=bin_u8, transform=transform):
        if value != 1:
            continue
        feats.append({"type": "Feature", "geometry": geom, "properties": {}})
    if not feats:
        return {"type": "FeatureCollection", "features": []}
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    utm = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm)
    gdf = gdf[gdf_utm.area / 10_000 >= float(min_area_ha)].reset_index(drop=True)
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    return json.loads(gdf.to_crs("EPSG:4326").to_json())
