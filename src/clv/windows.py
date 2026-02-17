import pandas as pd
from datetime import timedelta


def compute_windows(max_date, observation_days, gap_days, prediction_days):
    """
    Given dataset max date, compute observation, gap and prediction windows.
    """
    max_date = pd.to_datetime(max_date).normalize()
    
    prediction_end = max_date
    prediction_start = prediction_end - timedelta(days=prediction_days)

    gap_end = prediction_start
    gap_start = gap_end - timedelta(days=gap_days)

    observation_end = gap_start
    observation_start = observation_end - timedelta(days=observation_days)

    return {
        "observation_start": observation_start,
        "observation_end": observation_end,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "prediction_start": prediction_start,
        "prediction_end": prediction_end,
    }



def generate_cutoffs(min_date, max_date, observation_days, gap_days, prediction_days, step_days):
    """
    Generate cutoff dates such that all windows fit within [min_date, max_date].
    cutoff_date == observation_end
    """
    min_date = pd.to_datetime(min_date).normalize()
    max_date = pd.to_datetime(max_date).normalize()

    earliest_cutoff = min_date + timedelta(days=observation_days)
    latest_cutoff = max_date - timedelta(days=gap_days + prediction_days)

    if earliest_cutoff > latest_cutoff:
        return []

    cutoffs = []
    c = earliest_cutoff
    while c <= latest_cutoff:
        cutoffs.append(c)
        c += timedelta(days=step_days)

    return cutoffs

def compute_windows_from_cutoff(cutoff_date, observation_days, gap_days, prediction_days):
    cutoff_date = pd.to_datetime(cutoff_date).normalize()

    observation_end = cutoff_date
    observation_start = observation_end - timedelta(days=observation_days)

    gap_start = observation_end
    gap_end = gap_start + timedelta(days=gap_days)

    prediction_start = gap_end
    prediction_end = prediction_start + timedelta(days=prediction_days)

    return {
        "observation_start": observation_start,
        "observation_end": observation_end,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "prediction_start": prediction_start,
        "prediction_end": prediction_end,
    }
