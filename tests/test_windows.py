from datetime import date

from src.clv.windows import compute_windows_from_cutoff, generate_cutoffs


def test_generate_cutoffs_count():
    cutoffs = generate_cutoffs(date(2020, 1, 1), date(2020, 8, 1), 60, 14, 30, 20)
    assert len(cutoffs) > 0


def test_compute_windows_order():
    w = compute_windows_from_cutoff(date(2020, 5, 1), 60, 14, 30)
    assert w["obs_start"] < w["obs_end"] <= w["gap_start"] < w["gap_end"] <= w["pred_start"] < w["pred_end"]
