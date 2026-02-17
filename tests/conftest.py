from __future__ import annotations

from pathlib import Path

import pytest

from scripts.seed_demo_data import seed


@pytest.fixture()
def test_config(tmp_path: Path):
    csv = tmp_path / "sample.csv"
    db = tmp_path / "test.duckdb"
    seed(str(csv), n_customers=40, days=200)
    return {
        "data": {
            "db_path": str(db),
            "input_csv": str(csv),
            "observation_days": 60,
            "gap_days": 14,
            "prediction_days": 30,
        },
        "rolling": {"enabled": True, "step_days": 20},
        "models": {
            "churn": {"xgb_params": {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1, "eval_metric": "logloss"}, "calibrate": False},
            "revenue": {
                "spend_logreg": {"C": 1.0, "max_iter": 200, "random_state": 42},
                "revenue_reg": {"max_depth": 3, "learning_rate": 0.1, "random_state": 42},
            },
        },
        "decisioning": {
            "default_budget_eur": 100,
            "default_cost_per_customer": 1,
            "default_max_customers": 200,
            "default_save_rate": 0.2,
            "default_w_loss": 0.7,
            "default_w_clv": 0.3,
        },
    }
