"""Decode the three weak-label encodings used by the challenge.

Each parser returns a ``(confidence, days_since_1970)`` pair where

* ``confidence`` is a ``float32`` array in ``[0, 1]`` (0 = no alert).
* ``days_since_1970`` is an ``int32`` array holding a **UNIX days epoch**.
  Using the same epoch for every source makes post-hoc fusion trivial.
  Value ``0`` means "no alert for this pixel".

Only alerts on or after ``min_date`` (default 2020-01-01) are kept, per the
challenge definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np


_UNIX_EPOCH = date(1970, 1, 1)
_RADD_EPOCH = date(2014, 12, 31)
_GLADS2_EPOCH = date(2019, 1, 1)

DEFAULT_MIN_DATE = date(2020, 1, 1)


def _to_unix_days(d: date) -> int:
    return (d - _UNIX_EPOCH).days


@dataclass
class WeakLabel:
    """Parsed weak label source aligned to a single grid."""

    confidence: np.ndarray  # (H, W) float32 in [0, 1]
    days: np.ndarray        # (H, W) int32, UNIX days (0 = no alert)

    @property
    def mask(self) -> np.ndarray:
        return self.confidence > 0


# --- RADD -------------------------------------------------------------------


def parse_radd(
    raster: np.ndarray,
    *,
    score_low: float = 0.6,
    score_high: float = 1.0,
    min_date: date = DEFAULT_MIN_DATE,
) -> WeakLabel:
    """Decode RADD: leading digit=confidence (2 low, 3 high); rest=days since 2014-12-31.

    Examples from the challenge notebook:
      ``20001`` → low-conf on 2015-01-01
      ``30055`` → high-conf on 2015-02-24
    """
    r = raster.astype(np.int64, copy=False)
    conf = np.zeros(r.shape, dtype=np.float32)
    days_out = np.zeros(r.shape, dtype=np.int32)

    # Identify valid (non-zero) alert pixels.
    valid = r > 0
    if not np.any(valid):
        return WeakLabel(conf, days_out)

    # Leading digit is 10^k where k = floor(log10(v)).
    # Only values with leading digit 2 or 3 are meaningful.
    lead = np.zeros_like(r)
    lead[valid] = _leading_digit(r[valid])
    remainder = np.zeros_like(r)
    remainder[valid] = _strip_leading_digit(r[valid])

    # Days-since-2014-12-31 → UNIX days.
    offset = _to_unix_days(_RADD_EPOCH)
    unix_days = np.zeros_like(r, dtype=np.int32)
    unix_days[valid] = remainder[valid].astype(np.int32) + offset

    min_unix = _to_unix_days(min_date)
    keep = valid & (unix_days >= min_unix) & ((lead == 2) | (lead == 3))

    conf[keep & (lead == 2)] = score_low
    conf[keep & (lead == 3)] = score_high
    days_out[keep] = unix_days[keep]

    return WeakLabel(conf, days_out)


def _power10(x: np.ndarray) -> np.ndarray:
    """Largest 10^k <= x, for positive ints. Vectorized, int64."""
    y = x.astype(np.int64, copy=False)
    p = np.ones_like(y)
    # RADD alerts have at most 5-6 digits (days since 2014-12-31 ≤ ~10k),
    # leading digit at 10^4 for 2xxxx / 3xxxx encodings. Loop a few times.
    for _ in range(8):
        mask = y >= p * 10
        if not np.any(mask):
            break
        p[mask] *= 10
    return p


def _leading_digit(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.int64, copy=False)
    return (y // _power10(y)).astype(np.int64)


def _strip_leading_digit(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.int64, copy=False)
    p = _power10(y)
    return y - (y // p) * p


# --- GLAD-L -----------------------------------------------------------------


def parse_gladl(
    alert_raster: np.ndarray,
    date_raster: np.ndarray,
    *,
    yy: int,
    score_prob: float = 0.5,
    score_conf: float = 0.9,
    min_date: date = DEFAULT_MIN_DATE,
) -> WeakLabel:
    """Decode one GLAD-L year of (alert, alertDate) rasters.

    alert: 0=none, 2=probable, 3=confirmed.
    alertDate: day-of-year within 20YY; 0=no alert.
    """
    a = alert_raster.astype(np.uint8, copy=False)
    d_doy = date_raster.astype(np.int32, copy=False)

    conf = np.zeros(a.shape, dtype=np.float32)
    days = np.zeros(a.shape, dtype=np.int32)

    year = 2000 + int(yy)
    jan1_unix = _to_unix_days(date(year, 1, 1))

    valid = (a > 0) & (d_doy > 0)
    if not np.any(valid):
        return WeakLabel(conf, days)

    # DOY is 1-indexed; UNIX days of DOY=n is Jan1_unix + (n-1).
    unix_days = np.zeros_like(d_doy, dtype=np.int32)
    unix_days[valid] = jan1_unix + d_doy[valid] - 1

    min_unix = _to_unix_days(min_date)
    keep = valid & (unix_days >= min_unix)

    conf[keep & (a == 2)] = score_prob
    conf[keep & (a == 3)] = score_conf
    days[keep] = unix_days[keep]

    return WeakLabel(conf, days)


# --- GLAD-S2 ----------------------------------------------------------------


def parse_glads2(
    alert_raster: np.ndarray,
    date_raster: np.ndarray,
    *,
    scores: dict[int, float] | None = None,
    min_date: date = DEFAULT_MIN_DATE,
) -> WeakLabel:
    """Decode GLAD-S2 (alert, alertDate) rasters.

    alert: 0=none, 1=recent, 2=low, 3=medium, 4=high.
    alertDate: days since 2019-01-01; 0=no alert.
    """
    if scores is None:
        scores = {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}

    a = alert_raster.astype(np.uint8, copy=False)
    d_off = date_raster.astype(np.int32, copy=False)

    conf = np.zeros(a.shape, dtype=np.float32)
    days = np.zeros(a.shape, dtype=np.int32)

    valid = (a > 0) & (d_off > 0)
    if not np.any(valid):
        return WeakLabel(conf, days)

    offset = _to_unix_days(_GLADS2_EPOCH)
    unix_days = np.zeros_like(d_off, dtype=np.int32)
    unix_days[valid] = d_off[valid] + offset

    min_unix = _to_unix_days(min_date)
    keep = valid & (unix_days >= min_unix)

    for level, s in scores.items():
        conf[keep & (a == level)] = s
    days[keep & (conf > 0)] = unix_days[keep & (conf > 0)]

    return WeakLabel(conf, days)


# --- Date ↔ YYMM helper -----------------------------------------------------


def unix_days_to_yymm(unix_days: int) -> int:
    """Convert UNIX-day count to YYMM (e.g. 2021-04 → 2104)."""
    if unix_days <= 0:
        return 0
    d = _UNIX_EPOCH + timedelta(days=int(unix_days))
    return (d.year % 100) * 100 + d.month


def datetime_to_unix_days(d: datetime | date) -> int:
    if isinstance(d, datetime):
        d = d.date()
    return _to_unix_days(d)
