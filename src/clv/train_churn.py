from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from .config import ROOT
from .db import get_connection
from .metrics import binary_metrics


def _split_by_cutoff(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoffs = sorted(df["cutoff_date"].unique())
    split_idx = max(1, int(len(cutoffs) * 0.8))
    train_cutoffs = set(cutoffs[:split_idx])
    return df[df["cutoff_date"].isin(train_cutoffs)], df[~df["cutoff_date"].isin(train_cutoffs)]


def train_churn_model(config: dict) -> tuple[object, list[str], dict]:
    con = get_connection(config["data"]["db_path"])
    df = con.execute("SELECT * FROM customer_model_data_rollup").fetch_df()
    con.close()

    train_df, test_df = _split_by_cutoff(df)
    drop_cols = {"cutoff_date", "CustomerID", "churn_label", "revenue_pred_window"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    x_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep_cols = x_train.columns[x_train.notna().any()].tolist()
    x_train = x_train[keep_cols].fillna(x_train.median(numeric_only=True))
    y_train = train_df["churn_label"].astype(int)

    x_test = (
        test_df[keep_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(x_train.median())
    )
    y_test = test_df["churn_label"].astype(int)

    model = XGBClassifier(**config["models"]["churn"]["xgb_params"])
    model.fit(x_train, y_train)
    if config["models"]["churn"].get("calibrate", True):
        model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        model.fit(x_train, y_train)

    probs = model.predict_proba(x_test)[:, 1] if len(x_test) else np.array([0.5])
    metrics = binary_metrics(y_test if len(y_test) else [0], probs)

    model_dir = ROOT / "artifacts/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "churn_xgb.joblib")
    joblib.dump(keep_cols, model_dir / "feature_cols.joblib")

    try:
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.Explainer(model.predict_proba, x_train.iloc[:200])
        values = explainer(x_test.iloc[:50])
        shap.summary_plot(values[:, :, 1], x_test.iloc[:50], show=False)
        plt.tight_layout()
        plt.savefig(ROOT / "artifacts/reports/shap_churn_summary.png")
        plt.close()
    except Exception:
        pass

    return model, keep_cols, metrics
