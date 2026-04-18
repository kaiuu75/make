from datetime import date

import numpy as np

from deforest.labels.parsers import (
    parse_gladl,
    parse_glads2,
    parse_radd,
    unix_days_to_yymm,
)


def test_radd_high_confidence_post_2020():
    # 30055 → high-conf on 2015-02-24; 31847 → high-conf on 2020-01-21; 32300 → 2021-04-17
    raster = np.array([[0, 30055, 31847, 32300]], dtype=np.int32)
    out = parse_radd(raster, min_date=date(2020, 1, 1))
    # Only the last two survive; only the last has leading digit 3 and date ≥ 2020
    assert np.array_equal(out.mask, [[False, False, True, True]])
    assert out.confidence[0, 3] == 1.0  # high
    assert out.confidence[0, 2] == 1.0  # high


def test_radd_leading_digit_low():
    raster = np.array([[21847]], dtype=np.int32)  # low-confidence 2020-01-21
    out = parse_radd(raster, min_date=date(2020, 1, 1))
    assert out.confidence[0, 0] == 0.6


def test_radd_drops_pre_2020():
    raster = np.array([[20001, 30055]], dtype=np.int32)  # 2015 dates
    out = parse_radd(raster, min_date=date(2020, 1, 1))
    assert not out.mask.any()


def test_gladl_year_22():
    alert = np.array([[0, 2, 3]], dtype=np.uint8)
    day_of_year = np.array([[0, 100, 200]], dtype=np.uint16)
    out = parse_gladl(alert, day_of_year, yy=22)
    assert np.array_equal(out.mask, [[False, True, True]])
    assert out.confidence[0, 1] == 0.5   # probable
    assert out.confidence[0, 2] == 0.9   # confirmed
    # 2022-04-10 → YYMM 2204
    yymm = unix_days_to_yymm(int(out.days[0, 1]))
    assert yymm == 2204


def test_glads2_levels():
    alert = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
    days_off = np.array([[0, 500, 500, 500, 500]], dtype=np.uint16)  # 2020-05-15
    out = parse_glads2(alert, days_off)
    assert out.confidence[0, 0] == 0.0
    assert out.confidence[0, 1] == 0.25
    assert out.confidence[0, 4] == 1.0


def test_unix_days_to_yymm_roundtrip():
    from deforest.labels.parsers import datetime_to_unix_days

    for y, m, d, expected in [
        (2022, 4, 15, 2204),
        (2021, 12, 1, 2112),
        (2020, 1, 5, 2001),
    ]:
        assert unix_days_to_yymm(datetime_to_unix_days(date(y, m, d))) == expected
