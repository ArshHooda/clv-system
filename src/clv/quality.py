from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def validate_prediction_store(con, latest_table_or_view):
    cols = {r[1] for r in con.execute(f"PRAGMA table_info('{latest_table_or_view}')").fetchall()}
    required = {"CustomerID", "churn_prob", "expected_revenue", "expected_clv", "expected_loss"}
    miss = required - cols
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    dup = con.execute(
        f"SELECT COUNT(*) FROM (SELECT CustomerID, COUNT(*) c FROM {latest_table_or_view} GROUP BY 1 HAVING c > 1)"
    ).fetchone()[0]
    if dup:
        raise ValueError("Duplicate CustomerID in latest predictions")
    neg = con.execute(
        f"SELECT COUNT(*) FROM {latest_table_or_view} WHERE expected_revenue < 0 OR expected_clv < 0 OR expected_loss < 0"
    ).fetchone()[0]
    if neg:
        raise ValueError("Negative predicted values")


def _psi(expected, actual, bins=10):
    breaks = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    e_hist, _ = np.histogram(expected, bins=breaks)
    a_hist, _ = np.histogram(actual, bins=breaks)
    e_pct = np.where(e_hist == 0, 1e-6, e_hist / len(expected))
    a_pct = np.where(a_hist == 0, 1e-6, a_hist / len(actual))
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def data_drift_report(train_df, latest_df):
    feats = [c for c in ["net_revenue_obs", "txn_count_obs", "recency_days_obs"] if c in train_df.columns]
    return {f: _psi(train_df[f].fillna(0), latest_df[f].fillna(0)) for f in feats}


def calibration_check(y_true, probs):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    return {"brier": float(brier_score_loss(y_true, probs)), "curve": {"frac_pos": frac_pos.tolist(), "mean_pred": mean_pred.tolist()}}


def threshold_sanity(probs):
    arr = np.asarray(probs)
    return {
        "quantiles": {str(q): float(np.quantile(arr, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
        "pct_above_0.7": float((arr > 0.7).mean()),
        "pct_above_0.5": float((arr > 0.5).mean()),
    }
