"""Local metrics matching the leaderboard definitions.

Given a *predicted* GeoJSON FeatureCollection and a *ground-truth* one (both
``EPSG:4326``), compute:

* ``union_iou`` — area of the intersection of the global predicted union and
  the global ground-truth union, divided by the area of their union. **This
  is the main leaderboard score.**
* ``polygon_recall`` — fraction of GT polygons that have *any* intersection
  with a predicted polygon.
* ``polygon_level_fpr`` — fraction of predicted polygons that have *no*
  intersection with any GT polygon. (False positive rate at polygon level.)
* ``year_accuracy`` — among GT polygons matched to a predicted polygon,
  fraction for which ``prediction.time_step // 100 == gt.time_step // 100``
  (i.e. predicted year equals ground-truth year). Undefined if there are no
  matches or if the prediction omits ``time_step`` everywhere.

All areas are computed after projecting to the estimated UTM CRS so the
result is metric-accurate regardless of latitude.
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
    polygon_level_fpr: float
    year_accuracy: float | None
    n_pred: int
    n_gt: int
    n_matched: int

    def as_dict(self) -> dict:
        return {
            "union_iou": self.union_iou,
            "polygon_recall": self.polygon_recall,
            "polygon_level_fpr": self.polygon_level_fpr,
            "year_accuracy": self.year_accuracy,
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "n_matched": self.n_matched,
        }


def evaluate(pred_fc: dict, gt_fc: dict) -> EvalResult:
    pred = _load(pred_fc)
    gt = _load(gt_fc)

    # Project both to a common UTM for metric-accurate areas.
    if pred.empty and gt.empty:
        return EvalResult(1.0, 1.0, 0.0, None, 0, 0, 0)
    src = pred if not pred.empty else gt
    utm = src.estimate_utm_crs()
    pred_u = pred.to_crs(utm) if not pred.empty else pred
    gt_u = gt.to_crs(utm) if not gt.empty else gt

    union_iou = _union_iou(pred_u, gt_u)
    recall, matched_gt = _polygon_recall(pred_u, gt_u)
    fpr = _polygon_level_fpr(pred_u, gt_u)
    year_acc = _year_accuracy(pred_u, gt_u, matched_gt)

    return EvalResult(
        union_iou=union_iou,
        polygon_recall=recall,
        polygon_level_fpr=fpr,
        year_accuracy=year_acc,
        n_pred=len(pred_u),
        n_gt=len(gt_u),
        n_matched=sum(matched_gt),
    )


# --- helpers ---------------------------------------------------------------


def _load(fc: dict) -> gpd.GeoDataFrame:
    feats = fc.get("features", []) or []
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    geoms = [shape(f["geometry"]) for f in feats]
    props = [f.get("properties", {}) or {} for f in feats]
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")
    return gdf


def _union_iou(pred: gpd.GeoDataFrame, gt: gpd.GeoDataFrame) -> float:
    p_union = unary_union(pred.geometry.values) if len(pred) else None
    g_union = unary_union(gt.geometry.values) if len(gt) else None
    if p_union is None and g_union is None:
        return 1.0
    if p_union is None or g_union is None:
        return 0.0
    inter = p_union.intersection(g_union).area
    union = p_union.union(g_union).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def _polygon_recall(
    pred: gpd.GeoDataFrame, gt: gpd.GeoDataFrame
) -> tuple[float, list[bool]]:
    if gt.empty:
        return 1.0, []
    if pred.empty:
        return 0.0, [False] * len(gt)
    p_union = unary_union(pred.geometry.values)
    matched = [g.intersects(p_union) for g in gt.geometry.values]
    if len(matched) == 0:
        return 1.0, matched
    return float(sum(matched) / len(matched)), matched


def _polygon_level_fpr(pred: gpd.GeoDataFrame, gt: gpd.GeoDataFrame) -> float:
    if pred.empty:
        return 0.0
    if gt.empty:
        return 1.0
    g_union = unary_union(gt.geometry.values)
    fp = sum(1 for p in pred.geometry.values if not p.intersects(g_union))
    return float(fp / len(pred))


def _year_accuracy(
    pred: gpd.GeoDataFrame, gt: gpd.GeoDataFrame, matched_gt: list[bool]
) -> float | None:
    if "time_step" not in pred.columns or "time_step" not in gt.columns:
        return None
    gt_m = gt[matched_gt].reset_index(drop=True)
    if gt_m.empty:
        return None
    # For each matched GT, find any intersecting predicted polygon's time_step.
    correct = 0
    total = 0
    for i, g in enumerate(gt_m.geometry.values):
        gt_ts = gt_m.iloc[i].get("time_step")
        if gt_ts is None:
            continue
        gt_year = int(gt_ts) // 100
        candidates = pred[pred.geometry.intersects(g)]
        if candidates.empty:
            continue
        # Take time_step with the largest intersection area.
        best_ts = None
        best_area = -1.0
        for _, row in candidates.iterrows():
            inter = row.geometry.intersection(g).area
            if inter > best_area and row.get("time_step") is not None:
                best_area = inter
                best_ts = int(row["time_step"])
        if best_ts is None:
            continue
        total += 1
        if best_ts // 100 == gt_year:
            correct += 1
    if total == 0:
        return None
    return float(correct / total)
