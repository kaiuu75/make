"""Decode the three weak-label encodings used by the challenge.

Encodings (from ``makeathon26/challenge.ipynb`` §5):

- RADD: one integer combines confidence and date.
    leading digit 2 = low / 3 = high; remaining digits = days since 2014-12-31.
- GLAD-L: per-year ``(alertYY, alertDateYY)`` rasters.
    alertYY ∈ {0, 2=probable, 3=confirmed}; alertDate = DOY within 20YY.
- GLAD-S2: single ``(alert, alertDate)`` raster covering all years.
    alert ∈ {0, 1=recent, 2=low, 3=medium, 4=high};
    alertDate = days since 2019-01-01.

Each parser returns a :class:`WeakLabel` with

* ``confidence`` — float32 in ``[0, 1]`` (0 = no alert).
* ``days``      — int32 **UNIX days** (since 1970-01-01), 0 = no alert.

Only alerts on or after ``min_date`` (default 2020-01-01) are kept, per the
challenge definition of deforestation.
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
    r = raster.astype(np.int64, copy=False)
    conf = np.zeros(r.shape, dtype=np.float32)
    days_out = np.zeros(r.shape, dtype=np.int32)

    valid = r > 0
    if not np.any(valid):
        return WeakLabel(conf, days_out)

    lead = np.zeros_like(r)
    lead[valid] = _leading_digit(r[valid])
    remainder = np.zeros_like(r)
    remainder[valid] = _strip_leading_digit(r[valid])

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
    y = x.astype(np.int64, copy=False)
    p = np.ones_like(y)
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
    a = alert_raster.astype(np.uint8, copy=False)
    d_doy = date_raster.astype(np.int32, copy=False)

    conf = np.zeros(a.shape, dtype=np.float32)
    days = np.zeros(a.shape, dtype=np.int32)

    year = 2000 + int(yy)
    jan1_unix = _to_unix_days(date(year, 1, 1))

    valid = (a > 0) & (d_doy > 0)
    if not np.any(valid):
        return WeakLabel(conf, days)

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
    if unix_days <= 0:
        return 0
    d = _UNIX_EPOCH + timedelta(days=int(unix_days))
    return (d.year % 100) * 100 + d.month


def datetime_to_unix_days(d: datetime | date) -> int:
    if isinstance(d, datetime):
        d = d.date()
    return _to_unix_days(d)


def days_to_yymm_vectorized(days: np.ndarray) -> np.ndarray:
    """Vectorised UNIX-days → YYMM conversion."""
    out = np.zeros(days.shape, dtype=np.int32)
    valid = days > 0
    if not np.any(valid):
        return out

    epoch = date(1970, 1, 1)
    yr_start = np.array(
        [(date(y, 1, 1) - epoch).days for y in range(1970, 2101)],
        dtype=np.int64,
    )
    d = days[valid].astype(np.int64)
    idx = np.clip(np.searchsorted(yr_start, d, side="right") - 1, 0, len(yr_start) - 1)
    year = 1970 + idx
    doy = d - yr_start[idx]

    dom_norm = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
    dom_leap = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    month = np.empty_like(year)
    if np.any(~leap):
        month[~leap] = np.searchsorted(dom_norm, doy[~leap], side="right")
    if np.any(leap):
        month[leap] = np.searchsorted(dom_leap, doy[leap], side="right")
    month = np.clip(month, 1, 12)

    out[valid] = ((year % 100) * 100 + month).astype(np.int32)
    return out
