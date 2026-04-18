from deforest.evaluation.metrics import evaluate


def _fc(polys, time_steps=None):
    features = []
    for i, coords in enumerate(polys):
        props = {}
        if time_steps is not None:
            props["time_step"] = time_steps[i]
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


SQUARE_A = [[0, 0], [0, 0.01], [0.01, 0.01], [0.01, 0], [0, 0]]
SQUARE_B = [[0.005, 0.005], [0.005, 0.015], [0.015, 0.015], [0.015, 0.005], [0.005, 0.005]]
SQUARE_C = [[1, 1], [1, 1.01], [1.01, 1.01], [1.01, 1], [1, 1]]


def test_exact_match():
    res = evaluate(_fc([SQUARE_A]), _fc([SQUARE_A]))
    assert res.union_iou > 0.999
    assert res.polygon_recall == 1.0
    assert res.polygon_level_fpr == 0.0


def test_no_overlap():
    res = evaluate(_fc([SQUARE_A]), _fc([SQUARE_C]))
    assert res.union_iou == 0.0
    assert res.polygon_recall == 0.0
    assert res.polygon_level_fpr == 1.0


def test_partial_overlap_iou_between_0_and_1():
    res = evaluate(_fc([SQUARE_A]), _fc([SQUARE_B]))
    assert 0 < res.union_iou < 1
    assert res.polygon_recall == 1.0
    assert res.polygon_level_fpr == 0.0


def test_year_accuracy():
    res = evaluate(_fc([SQUARE_A], [2204]), _fc([SQUARE_A], [2207]))
    assert res.year_accuracy == 1.0  # same year, different month
    res2 = evaluate(_fc([SQUARE_A], [2204]), _fc([SQUARE_A], [2304]))
    assert res2.year_accuracy == 0.0


def test_empty_prediction():
    res = evaluate({"type": "FeatureCollection", "features": []}, _fc([SQUARE_A]))
    assert res.polygon_recall == 0.0
    assert res.union_iou == 0.0
