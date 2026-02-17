import duckdb
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def score_and_write_to_db(
    model,
    feature_cols: list[str],
    table_in: str = "customer_model_data_rollup",
    table_out: str = "predictions_customer",
):
    """
    Legacy churn-only scoring (kept for backward compatibility).
    """
    con = duckdb.connect("data/warehouse.duckdb")
    df = con.execute(f"SELECT * FROM {table_in}").fetchdf()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        con.close()
        raise ValueError(f"Missing feature columns in {table_in}: {missing}")

    # Use EXACT feature columns from training (keeps order)
    X = df[feature_cols].copy()

    # Convert to numeric safely + handle inf
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    df["churn_prob"] = model.predict_proba(X)[:, 1]
    df["risk_score"] = df["churn_prob"] * df["revenue_pred_window"]

    con.execute(f"DROP TABLE IF EXISTS {table_out}")
    con.register(
        "pred_df",
        df[["cutoff_date", "CustomerID", "churn_prob", "revenue_pred_window", "risk_score"]],
    )
    con.execute(f"CREATE TABLE {table_out} AS SELECT * FROM pred_df")
    con.close()

    print(f"Scored and saved to DuckDB table: {table_out}")


import numpy as np
import duckdb
import pandas as pd


def score_clv_and_write_to_db(
    churn_model,
    spend_model,
    revenue_model,
    feature_cols: list[str],
    table_in: str = "customer_model_data_rollup",
    table_out_prefix: str = "predictions_customer",
):
    con = duckdb.connect("data/warehouse.duckdb")
    df = con.execute(f"SELECT * FROM {table_in}").fetchdf()

    # Ensure required columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        con.close()
        raise ValueError(f"Missing feature columns in {table_in}: {missing}")

    # Build X with exact feature list and order
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    churn_prob = churn_model.predict_proba(X)[:, 1]
    spend_prob = spend_model.predict_proba(X)[:, 1]

    # Revenue model predicts log1p(revenue | spend>0)
    pred_rev_log = revenue_model.predict(X)
    pred_rev_if_spend = np.expm1(pred_rev_log)
    pred_rev_if_spend = np.clip(pred_rev_if_spend, 0, None)

    expected_revenue = spend_prob * pred_rev_if_spend
    expected_clv = (1 - churn_prob) * expected_revenue
    expected_loss = churn_prob * expected_revenue

    out = pd.DataFrame({
        "cutoff_date": pd.to_datetime(df["cutoff_date"]).dt.date,
        "CustomerID": df["CustomerID"],
        "churn_prob": churn_prob,
        "spend_prob": spend_prob,
        "pred_revenue_if_spend": pred_rev_if_spend,
        "expected_revenue": expected_revenue,
        "expected_clv": expected_clv,
        "expected_loss": expected_loss
    })

    # =========================
    # Versioned write (B)
    # =========================
    cutoffs = sorted(out["cutoff_date"].unique())

    for c in cutoffs:
        suffix = str(c).replace("-", "_")  # YYYY_MM_DD
        table_name = f"{table_out_prefix}_{suffix}"

        subset = out[out["cutoff_date"] == c].copy()

        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("out_df", subset)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM out_df")

        print(f"Saved versioned predictions table: {table_name} (rows={len(subset)})")

    # latest view
    latest = max(cutoffs)
    latest_suffix = str(latest).replace("-", "_")
    latest_table = f"{table_out_prefix}_{latest_suffix}"

    con.execute(f"DROP VIEW IF EXISTS {table_out_prefix}_latest")
    con.execute(f"CREATE VIEW {table_out_prefix}_latest AS SELECT * FROM {latest_table}")
    print(f"Created view: {table_out_prefix}_latest -> {latest_table}")

    con.close()
