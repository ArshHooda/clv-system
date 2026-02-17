import duckdb 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from clv.explain import shap_global_local
from clv.score import save_model, score_and_write_to_db
from clv.business import retention_simulation
from sklearn.calibration import CalibratedClassifierCV



def train_churn_model():
    con = duckdb.connect("data/warehouse.duckdb")
    df = con.execute("SELECT * FROM customer_model_data_rollup").fetchdf()

    con.close()

    # Drop non-feature columns
    drop_cols = [
        "cutoff_date",
        "CustomerID",
        "first_purchase_date_obs",
        "last_purchase_date_obs",
        "churn_label",
        "revenue_pred_window",

        # Correlation pruning
        "txn_count_obs",
        "invoice_count_obs",
        "active_days_obs",
        "txn_count_90d",
        "gross_revenue_obs",
        "net_revenue_90d",
        "return_revenue_obs"
    ]

    X = df.drop(columns=drop_cols)
    y = df["churn_label"]

    print("\n=== Correlation (top pairs > 0.85) ===")

    corr_matrix = X.corr().abs()

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.85:
                high_corr_pairs.append(
                    (corr_matrix.columns[i],
                     corr_matrix.columns[j],
                     round(corr_matrix.iloc[i, j], 3))
                )

    for pair in high_corr_pairs:
        print(pair)

    nan_counts = X.isna().sum().sort_values(ascending=False)
    print("\nNaN counts (top 10):")
    print(nan_counts.head(10))

    # Train-test split
    # Sort by first purchase date
    df = df.sort_values("cutoff_date")
    cutoffs = sorted(df["cutoff_date"].unique())
    split = int(len(cutoffs) * 0.8)

    train_cutoffs = cutoffs[:split]
    test_cutoffs = cutoffs[split:]

    train_df = df[df["cutoff_date"].isin(train_cutoffs)]
    test_df = df[df["cutoff_date"].isin(test_cutoffs)]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["churn_label"]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["churn_label"]
    import numpy as np

    # Keep only numeric columns (drop datetime/object columns automatically)
    X_train = X_train.select_dtypes(include=["number"]).copy()
    X_test  = X_test.select_dtypes(include=["number"]).copy()

    # Replace inf/-inf with NaN (safety)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns that are all NaN in train (median imputer can't compute)
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print("Dropping all-NaN columns:", all_nan_cols)
        X_train.drop(columns=all_nan_cols, inplace=True)
        X_test.drop(columns=all_nan_cols, inplace=True, errors="ignore")

    # ✅ ADDED (necessary): freeze the exact feature list used for training & scoring
    feature_cols = X_train.columns.tolist()

    print("\nNumeric feature columns used:", len(X_train.columns))
    print("NaN counts in X_train (top 10):")
    print(X_train.isna().sum().sort_values(ascending=False).head(10))

    print("Training Logistic Regression (Pipeline: imputer+scaler+logreg)...")
    log_model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict_proba(X_test)[:, 1]

    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)
    
    print("Calibrating churn probabilities (sigmoid)...")
    cal_model = CalibratedClassifierCV(xgb_model, method="sigmoid", cv=3)
    cal_model.fit(X_train, y_train)

    # Use calibrated probs for evaluation
    y_pred_xgb = cal_model.predict_proba(X_test)[:, 1]
    print("XGB+Cal ROC-AUC:", round(roc_auc_score(y_test, y_pred_xgb), 4))
    print("XGB+Cal PR-AUC:", round(average_precision_score(y_test, y_pred_xgb), 4))

    # Save calibrated model instead of raw xgb
    save_model(cal_model, "artifacts/models/churn_xgb.joblib")


    # Metrics
    print("\n=== Evaluation ===")
    print("Logistic ROC-AUC:", round(roc_auc_score(y_test, y_pred_log), 4))
    print("Logistic PR-AUC:", round(average_precision_score(y_test, y_pred_log), 4))

    print("XGBoost ROC-AUC:", round(roc_auc_score(y_test, y_pred_xgb), 4))
    print("XGBoost PR-AUC:", round(average_precision_score(y_test, y_pred_xgb), 4))

    # =========================
    # Business Evaluation
    # =========================

    # Get full test set predictions
    test_df = test_df.copy()
    test_df["churn_prob"] = xgb_model.predict_proba(X_test)[:, 1]

    # Strategy A: churn probability ranking
    test_df = test_df.sort_values("churn_prob", ascending=False)

    total_revenue = test_df["revenue_pred_window"].sum()

    for pct in [0.1, 0.2, 0.3]:
        top_n = int(len(test_df) * pct)
        revenue_captured = test_df.head(top_n)["revenue_pred_window"].sum()
        print(f"\nStrategy A - Top {int(pct*100)}% revenue capture:",
              round(revenue_captured / total_revenue, 4))

    # Strategy B: risk-adjusted revenue
    test_df["risk_score"] = test_df["churn_prob"] * test_df["revenue_pred_window"]
    test_df = test_df.sort_values("risk_score", ascending=False)

    for pct in [0.1, 0.2, 0.3]:
        top_n = int(len(test_df) * pct)
        revenue_captured = test_df.head(top_n)["revenue_pred_window"].sum()
        print(f"\nStrategy B - Top {int(pct*100)}% revenue capture:",
              round(revenue_captured / total_revenue, 4))

    print("\n=== Logistic Coefficients ===")
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": log_model.named_steps["clf"].coef_[0]
    }).sort_values("coefficient", ascending=False)

    print(coef_df)

    print("\n=== XGBoost Feature Importance ===")
    xgb_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)

    # ✅ ADDED (necessary): you created it but never printed it
    print(xgb_importance)

    # Save churn model
    save_model(xgb_model, "artifacts/models/churn_xgb.joblib")

    # Build business frame for test set
    test_df = test_df.copy()
    test_df["churn_prob"] = xgb_model.predict_proba(X_test)[:, 1]
    test_df["risk_score"] = test_df["churn_prob"] * test_df["revenue_pred_window"]

    # Run one example simulation
    sim = retention_simulation(
        test_df[["CustomerID", "churn_prob", "revenue_pred_window", "risk_score"]],
        target_pct=0.2,
        save_rate=0.15,
        cost_per_customer=1.0,
    )
    print("\n=== Retention Simulation Example (Top 20%) ===")
    print({k: v for k, v in sim.items() if k != "target_list"})

    # SHAP artifacts
    shap_info = shap_global_local(xgb_model, X_train, X_test, test_df["CustomerID"])
    print("\nSHAP saved:", shap_info)

    # Score all rows and write back to DuckDB
    score_and_write_to_db(xgb_model, feature_cols=feature_cols)

    # ✅ ADDED (necessary): your __main__ expects these
    return log_model, xgb_model


if __name__ == "__main__":
    log_model, xgb_model = train_churn_model()

    from joblib import dump
    import os

    os.makedirs("artifacts/models", exist_ok=True)

    dump(log_model, "artifacts/models/churn_logistic.joblib")
    dump(xgb_model, "artifacts/models/churn_xgb.joblib")

    print("\nModels saved successfully.")
