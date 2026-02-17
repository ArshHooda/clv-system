import os
import duckdb
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from clv.score import save_model


DB_PATH = "data/warehouse.duckdb"
TABLE_IN = "customer_model_data_rollup"


def train_revenue_models() -> None:
    con = duckdb.connect(DB_PATH)
    df = con.execute(f"SELECT * FROM {TABLE_IN}").fetchdf()
    con.close()

    if df.empty:
        raise ValueError(f"{TABLE_IN} is empty. Run rolling build first.")

    # Same drop cols philosophy as train_churn.py (keep consistent)
    drop_cols = [
        "cutoff_date",
        "CustomerID",
        "first_purchase_date_obs",
        "last_purchase_date_obs",
        "churn_label",
        "revenue_pred_window",

        # optional redundancy
        "txn_count_obs",
        "invoice_count_obs",
        "active_days_obs",
        "txn_count_90d",
        "gross_revenue_obs",
        "net_revenue_90d",
        "return_revenue_obs",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Forward split by cutoff_date (same as churn)
    df = df.sort_values("cutoff_date")
    cutoffs = sorted(df["cutoff_date"].unique())
    if len(cutoffs) < 2:
        raise ValueError("Need at least 2 cutoffs for revenue training.")

    split = int(len(cutoffs) * 0.8)
    split = max(1, split)
    train_cutoffs = cutoffs[:split]
    test_cutoffs = cutoffs[split:]

    train_df = df[df["cutoff_date"].isin(train_cutoffs)].copy()
    test_df = df[df["cutoff_date"].isin(test_cutoffs)].copy()

    # Build X
    X_train = train_df.drop(columns=drop_cols).select_dtypes(include=["number"]).copy()
    X_test  = test_df.drop(columns=drop_cols).select_dtypes(include=["number"]).copy()

    # Safety
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print("Dropping all-NaN columns:", all_nan_cols)
        X_train.drop(columns=all_nan_cols, inplace=True)
        X_test.drop(columns=all_nan_cols, inplace=True, errors="ignore")

    feature_cols = X_train.columns.tolist()
    print("\nFeature cols used:", len(feature_cols))
    print("Train cutoffs:", train_cutoffs)
    print("Test cutoffs :", test_cutoffs)

    # ===== B1) Spend model: P(revenue > 0) =====
    y_spend_train = (train_df["revenue_pred_window"] > 0).astype(int)
    y_spend_test  = (test_df["revenue_pred_window"] > 0).astype(int)

    spend_model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    print("\nTraining Spend Classifier...")
    spend_model.fit(X_train, y_spend_train)
    spend_prob_test = spend_model.predict_proba(X_test)[:, 1]

    print("Spend ROC-AUC:", round(roc_auc_score(y_spend_test, spend_prob_test), 4))
    print("Spend PR-AUC :", round(average_precision_score(y_spend_test, spend_prob_test), 4))

    # ===== B2) Revenue model: E[rev | rev > 0] =====
    pos_mask_train = train_df["revenue_pred_window"] > 0
    pos_mask_test  = test_df["revenue_pred_window"] > 0

    X_train_pos = X_train.loc[pos_mask_train].copy()
    y_train_pos = train_df.loc[pos_mask_train, "revenue_pred_window"].astype(float)

    X_test_pos = X_test.loc[pos_mask_test].copy()
    y_test_pos = test_df.loc[pos_mask_test, "revenue_pred_window"].astype(float)

    # log transform for stability
    y_train_log = np.log1p(y_train_pos)

    rev_model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.08,
            random_state=42
        ))
    ])

    print("\nTraining Revenue Regressor (conditional on spend>0)...")
    rev_model.fit(X_train_pos, y_train_log)

    # Predict for positive test customers
    pred_log_pos = rev_model.predict(X_test_pos)
    pred_pos = np.expm1(pred_log_pos)  # back-transform
    pred_pos = np.clip(pred_pos, 0, None)

    print("Revenue MAE (pos only):", round(float(mean_absolute_error(y_test_pos, pred_pos)), 2))
    print("Revenue R2  (pos only):", round(float(r2_score(y_test_pos, pred_pos)), 4))

    # ===== Save models + feature list =====
    os.makedirs("artifacts/models", exist_ok=True)

    save_model(spend_model, "artifacts/models/spend_clf.joblib")
    save_model(rev_model, "artifacts/models/revenue_reg.joblib")
    save_model(feature_cols, "artifacts/models/feature_cols.joblib")

    print("\nSaved:")
    print("- artifacts/models/spend_clf.joblib")
    print("- artifacts/models/revenue_reg.joblib")
    print("- artifacts/models/feature_cols.joblib")


if __name__ == "__main__":
    train_revenue_models()
