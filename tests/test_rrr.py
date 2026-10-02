import math

from src.eval.rrr import compute_rrr

S = {
    ("base", "a"): 0.30, ("spec-a", "a"): 0.50,
    ("base", "b"): 0.20, ("spec-b", "b"): 0.60,
    ("base", "c"): 0.10, ("spec-c", "c"): 0.11,   # gain too small -> n/a
}


def test_rrr_values():
    s = {**S, ("m", "a"): 0.40, ("m", "b"): 0.30, ("m", "c"): 0.10}
    vals, macro, worst = compute_rrr(s, "m", ["a", "b", "c"])
    assert math.isclose(vals[0], 0.5) and math.isclose(vals[1], 0.25) and vals[2] is None
    assert math.isclose(macro, 0.375) and math.isclose(worst, 0.25)


def test_rrr_full_and_negative():
    s = {**S, ("m", "a"): 0.50, ("m", "b"): 0.10}
    vals, _, worst = compute_rrr(s, "m", ["a", "b"])
    assert math.isclose(vals[0], 1.0)            # kept full gain
    assert math.isclose(vals[1], -0.25)          # fell below base
    assert math.isclose(worst, -0.25)


def test_rrr_missing_merged_score():
    vals, macro, _ = compute_rrr(S, "m", ["a"])
    assert vals == [None] and math.isnan(macro)
