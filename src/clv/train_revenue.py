from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ROOT
from .db import get_connection
from .metrics import binary_metrics
from .train_churn import _split_by_cutoff


def train_revenue_models(config: dict) -> tuple[object, object, dict]:
    con = get_connection(config["data"]["db_path"])
    df = con.execute("SELECT * FROM customer_model_data_rollup").fetch_df()
    con.close()

    feature_cols = joblib.load(ROOT / "artifacts/models/feature_cols.joblib")
    train_df, test_df = _split_by_cutoff(df)
    x_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    y_spend_train = (train_df["revenue_pred_window"] > 0).astype(int)
    y_spend_test = (test_df["revenue_pred_window"] > 0).astype(int)

    spend = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**config["models"]["revenue"]["spend_logreg"])),
        ]
    )
    spend.fit(x_train, y_spend_train)
    spend_probs = spend.predict_proba(x_test)[:, 1] if len(x_test) else np.array([0.5])
    spend_metrics = binary_metrics(y_spend_test if len(y_spend_test) else [0], spend_probs)

    pos_train = train_df[train_df["revenue_pred_window"] > 0]
    pos_test = test_df[test_df["revenue_pred_window"] > 0]
    reg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(**config["models"]["revenue"]["revenue_reg"])),
        ]
    )
    reg.fit(pos_train[feature_cols], np.log1p(pos_train["revenue_pred_window"]))

    if len(pos_test):
        pred_pos = np.expm1(reg.predict(pos_test[feature_cols]))
        reg_metrics = {
            "mae": float(mean_absolute_error(pos_test["revenue_pred_window"], pred_pos)),
            "r2": float(r2_score(pos_test["revenue_pred_window"], pred_pos)),
        }
    else:
        reg_metrics = {"mae": 0.0, "r2": 0.0}

    joblib.dump(spend, ROOT / "artifacts/models/spend_clf.joblib")
    joblib.dump(reg, ROOT / "artifacts/models/revenue_reg.joblib")
    return spend, reg, {"spend": spend_metrics, "reg": reg_metrics}
