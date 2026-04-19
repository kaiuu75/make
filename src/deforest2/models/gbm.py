"""LightGBM-based per-pixel deforestation classifier with the improvements

* **(3A) Soft-label cross-entropy** — ``objective="cross_entropy"`` accepts
  float targets in ``[0, 1]`` (max-confidence from the weak-label fusion)
  instead of hard 0/1 labels.
* **(4) Early stopping on a held-out set** — if ``eval_X/eval_y`` are
  provided to :meth:`PixelGBM.fit`, training stops once the validation
  metric plateaus.
* **(7) Explicit class weights** — prevalence via ``scale_pos_weight``
  (binary objective only) is separate from the per-sample confidence /
  region weights computed upstream in ``train_gbm.py``.

:func:`subsample_pixels` implements **(3B)** — the *ignore band*: pixels
whose max-source-confidence falls between ``hard_negative_max`` and the
positive-fusion cutoff are dropped from training rather than silently
treated as negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class GBMConfig:
    objective: str = "cross_entropy"            # or "binary"
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 200
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    num_threads: int | None = None
    scale_pos_weight: float | None = None        # None -> do not set
    metric: Sequence[str] = field(
        default_factory=lambda: ("binary_logloss", "average_precision")
    )
    seed: int = 1337


class PixelGBM:
    """Thin wrapper around :class:`lightgbm.Booster` for per-pixel prediction."""

    def __init__(self, cfg: GBMConfig | None = None):
        self.cfg = cfg or GBMConfig()
        self._booster = None
        self._feature_names: list[str] | None = None
        self.best_iteration_: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        eval_X: np.ndarray | None = None,
        eval_y: np.ndarray | None = None,
        eval_w: np.ndarray | None = None,
    ) -> "PixelGBM":
        import lightgbm as lgb

        self._feature_names = list(feature_names) if feature_names else None

        params = {
            "objective": self.cfg.objective,
            "metric": list(self.cfg.metric),
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.cfg.num_leaves,
            "min_child_samples": self.cfg.min_child_samples,
            "feature_fraction": self.cfg.feature_fraction,
            "bagging_fraction": self.cfg.bagging_fraction,
            "bagging_freq": self.cfg.bagging_freq,
            "verbose": -1,
            "seed": self.cfg.seed,
        }
        if self.cfg.num_threads is not None:
            params["num_threads"] = int(self.cfg.num_threads)
        if self.cfg.objective == "binary" and self.cfg.scale_pos_weight is not None:
            params["scale_pos_weight"] = float(self.cfg.scale_pos_weight)

        # LightGBM accepts float targets natively for cross_entropy; clip to
        # [0, 1] to protect against numerical drift from upstream weights.
        y_train = np.clip(y.astype(np.float32), 0.0, 1.0)

        dtrain = lgb.Dataset(
            X, label=y_train, weight=weights, feature_name=self._feature_names,
            free_raw_data=False,
        )
        valid_sets = [dtrain]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=100)]
        if eval_X is not None and eval_y is not None:
            y_eval = np.clip(eval_y.astype(np.float32), 0.0, 1.0)
            dval = lgb.Dataset(
                eval_X, label=y_eval, weight=eval_w,
                feature_name=self._feature_names, reference=dtrain,
                free_raw_data=False,
            )
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=int(self.cfg.early_stopping_rounds),
                    first_metric_only=True,
                    verbose=False,
                )
            )

        self._booster = lgb.train(
            params,
            dtrain,
            num_boost_round=int(self.cfg.n_estimators),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration_ = getattr(self._booster, "best_iteration", None)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("PixelGBM.predict_proba called before fit/load")
        return self._booster.predict(X).astype(np.float32)

    def save(self, path: str | Path) -> Path:
        if self._booster is None:
            raise RuntimeError("Cannot save an untrained PixelGBM")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(p))
        return p

    def load(self, path: str | Path) -> "PixelGBM":
        import lightgbm as lgb

        self._booster = lgb.Booster(model_file=str(path))
        self._feature_names = self._booster.feature_name()
        return self


# ---------------------------------------------------------------------------
# Per-tile sampling (Improvements 1, 3B, 7)
# ---------------------------------------------------------------------------


@dataclass
class TileSample:
    """Indices and metadata for one tile, pre-sampling.

    Collected in the **first pass** of :func:`scripts.train_gbm.main` so we
    can allocate a *global stratified quota* across regions in a second pass.
    """

    tile_id: str
    region: str
    pos_idx: np.ndarray
    neg_idx: np.ndarray
    pos_conf: np.ndarray
    pos_agree: np.ndarray
    pos_days: np.ndarray


def select_candidate_pixels(
    binary: np.ndarray,
    max_confidence: np.ndarray,
    agree_count: np.ndarray,
    days: np.ndarray,
    forest: np.ndarray,
    *,
    hard_negative_max: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose candidate positive / negative pixels for one tile.

    Improvement (3B): negatives are restricted to ``max_confidence <= hard_negative_max``
    inside the forest mask. Pixels with ``hard_negative_max < max_confidence``
    but not flagged as positives are silently *ignored* during training.
    """
    if binary.shape != max_confidence.shape:
        raise ValueError("binary and max_confidence must share shape")
    h, w = binary.shape

    bin_flat = binary.reshape(-1).astype(np.uint8)
    conf_flat = max_confidence.reshape(-1).astype(np.float32)
    agree_flat = agree_count.reshape(-1).astype(np.uint8)
    days_flat = days.reshape(-1).astype(np.int32)
    forest_flat = (forest > 0).reshape(-1)

    pos_idx = np.flatnonzero((bin_flat > 0) & forest_flat)
    neg_idx = np.flatnonzero(
        (bin_flat == 0) & forest_flat & (conf_flat <= float(hard_negative_max))
    )

    pos_conf = conf_flat[pos_idx]
    pos_agree = agree_flat[pos_idx]
    pos_days = days_flat[pos_idx]
    return pos_idx, neg_idx, pos_conf, pos_agree, pos_days


def subsample_positives_stratified(
    pos_conf: np.ndarray,
    cap: int,
    *,
    n_strata: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Confidence-stratified subsample of positive pixels.

    Returns an ``int`` array of *positions* into ``pos_conf``. The caller
    applies this to the aligned ``pos_idx`` / ``pos_agree`` arrays with a
    simple slice — no dict lookups, no allocations.

    Each stratum contributes ``cap // n_strata`` pixels. Any stratum with
    fewer available pixels gives up its budget; the short count is made up
    by randomly sampling the remaining positives.
    """
    rng = rng or np.random.default_rng(0)
    n = int(pos_conf.size)
    if n == 0 or cap <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    if n_strata <= 1:
        return rng.choice(n, size=cap, replace=False).astype(np.int64)

    qs = np.linspace(0.0, 1.0, n_strata + 1)[1:-1]
    edges = np.quantile(pos_conf, qs)
    bucket = np.digitize(pos_conf, edges)

    per_bucket = cap // n_strata
    kept_parts: list[np.ndarray] = []
    short_count = 0
    for b in range(n_strata):
        pos_in_bucket = np.flatnonzero(bucket == b)
        if pos_in_bucket.size <= per_bucket:
            kept_parts.append(pos_in_bucket)
            short_count += per_bucket - pos_in_bucket.size
        else:
            kept_parts.append(rng.choice(pos_in_bucket, size=per_bucket, replace=False))

    kept = np.concatenate(kept_parts) if kept_parts else np.empty(0, dtype=np.int64)
    if short_count > 0 and kept.size < cap:
        mask = np.ones(n, dtype=bool)
        if kept.size:
            mask[kept] = False
        remaining = np.flatnonzero(mask)
        if remaining.size > 0:
            take = min(short_count, remaining.size, cap - kept.size)
            kept = np.concatenate([kept, rng.choice(remaining, size=take, replace=False)])

    return kept.astype(np.int64)


def compute_positive_weights(
    pos_conf: np.ndarray,
    pos_agree: np.ndarray,
) -> np.ndarray:
    """Positive-pixel weight = confidence · (1 + 0.25 · agreement).

    The same formula as ``make/``; kept here so training scripts stay small.
    """
    return (pos_conf * (1.0 + 0.25 * pos_agree.astype(np.float32))).astype(np.float32)
