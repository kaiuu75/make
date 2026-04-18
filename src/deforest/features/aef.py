"""AlphaEarth Foundations feature builder.

Produces per-pixel features from annual AEF embeddings:

* ``aef_2020``     — (64, H, W)
* ``aef_last``     — (64, H, W) for the most recent available year
* ``aef_delta``    — (64, H, W) = last − 2020
* ``aef_norm``     — ||delta||₂ (H, W)  (useful as an interpretable signal)

All AEF arrays are expected to already be reprojected onto the S2 grid.
"""

from __future__ import annotations

import numpy as np


def aef_features(
    aef_by_year: dict[int, np.ndarray],
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

    return {
        "aef_base": base,       # (64, H, W)
        "aef_last": last,       # (64, H, W)
        "aef_delta": delta,     # (64, H, W)
        "aef_norm": norm,       # (H, W)
        "year_base": np.int32(year_base),
        "year_last": np.int32(year_last),
    }


def flatten_aef_for_pixels(feats: dict[str, np.ndarray]) -> np.ndarray:
    """Stack AEF feature arrays into (H*W, F) for per-pixel ML."""
    c = feats["aef_base"].shape[0]
    h, w = feats["aef_base"].shape[1:]
    stacked = np.concatenate(
        [
            feats["aef_base"].reshape(c, -1).T,
            feats["aef_last"].reshape(c, -1).T,
            feats["aef_delta"].reshape(c, -1).T,
            feats["aef_norm"].reshape(-1, 1),
        ],
        axis=1,
    )
    return np.nan_to_num(stacked, nan=0.0).astype(np.float32)
