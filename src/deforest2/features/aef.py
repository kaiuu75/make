"""AlphaEarth Foundations feature builder.

On top of the standard ``(base, last, delta, norm)`` features used in
``make/``, we also compute — when more than two AEF years are available —

* ``aef_max_drift``     — per-pixel maximum ``||aef(y) − aef(base)||₂`` across
  all years ``y > year_base``. This catches mid-range changes that happened
  and partially recovered by ``year_last``.
* ``aef_year_of_drift`` — the year in which the drift peaked (int16). Useful
  both as a feature and as a date proxy for ``time_step``.
* ``aef_cos_dist``      — cosine distance between base and last embeddings
  per pixel; scale-invariant and often a stronger signal than raw Δ dims.

All arrays are assumed to already be reprojected onto the S2 grid.
"""

from __future__ import annotations

import numpy as np


def aef_features(
    aef_by_year: dict[int, np.ndarray],
    *,
    multi_year_drift: bool = True,
) -> dict[str, np.ndarray]:
    """Return a dict of feature arrays from a ``{year: (64, H, W)}`` mapping."""
    if not aef_by_year:
        raise ValueError("aef_by_year must contain at least one year")

    years = sorted(aef_by_year)
    year_base = years[0]
    year_last = years[-1]

    base = aef_by_year[year_base].astype(np.float32)
    last = aef_by_year[year_last].astype(np.float32)

    delta = last - base
    norm = np.linalg.norm(np.nan_to_num(delta, nan=0.0), axis=0)

    cos_dist = _pixelwise_cosine_distance(base, last)

    out: dict[str, np.ndarray] = {
        "aef_base": base,
        "aef_last": last,
        "aef_delta": delta,
        "aef_norm": norm.astype(np.float32),
        "aef_cos_dist": cos_dist.astype(np.float32),
        "year_base": np.int32(year_base),
        "year_last": np.int32(year_last),
    }

    if multi_year_drift and len(years) >= 2:
        # Stack ||aef(y) - aef(base)|| across all non-base years and take the
        # per-pixel maximum + year-of-max.
        later = [y for y in years if y != year_base]
        if later:
            drifts = np.stack(
                [
                    np.linalg.norm(
                        np.nan_to_num(aef_by_year[y].astype(np.float32) - base, nan=0.0),
                        axis=0,
                    )
                    for y in later
                ],
                axis=0,
            )
            max_drift = drifts.max(axis=0).astype(np.float32)
            idx = drifts.argmax(axis=0)
            year_of_max = np.array(later, dtype=np.int16)[idx]
            out["aef_max_drift"] = max_drift
            out["aef_year_of_drift"] = year_of_max.astype(np.int16)

    return out


def _pixelwise_cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``1 - cos(a, b)`` along axis 0 for ``(C, H, W)`` tensors."""
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    num = (a * b).sum(axis=0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-6
    cos = num / den
    return (1.0 - cos).astype(np.float32)
