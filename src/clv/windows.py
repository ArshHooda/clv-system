from __future__ import annotations

from datetime import date, timedelta


def compute_windows_from_cutoff(cutoff_date: date, obs_days: int, gap_days: int, pred_days: int) -> dict:
    obs_end = cutoff_date
    obs_start = cutoff_date - timedelta(days=obs_days)
    gap_start = obs_end
    gap_end = gap_start + timedelta(days=gap_days)
    pred_start = gap_end
    pred_end = pred_start + timedelta(days=pred_days)
    return {
        "cutoff_date": cutoff_date,
        "obs_start": obs_start,
        "obs_end": obs_end,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "pred_start": pred_start,
        "pred_end": pred_end,
    }


def generate_cutoffs(
    min_date: date,
    max_date: date,
    obs_days: int,
    gap_days: int,
    pred_days: int,
    step_days: int,
) -> list[date]:
    first = min_date + timedelta(days=obs_days)
    last = max_date - timedelta(days=gap_days + pred_days)
    cutoffs: list[date] = []
    current = first
    while current <= last:
        cutoffs.append(current)
        current += timedelta(days=step_days)
    return cutoffs
