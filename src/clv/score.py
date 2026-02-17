from __future__ import annotations

import numpy as np
import pandas as pd

from .db import get_connection


def score_clv_and_write_to_db(
    churn_model,
    spend_model,
    revenue_model,
    feature_cols: list[str],
    table_in: str,
    table_out_prefix: str,
    db_path: str,
) -> str:
    con = get_connection(db_path)
    df = con.execute(f"SELECT * FROM {table_in}").fetch_df()

    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))

    churn_prob = churn_model.predict_proba(x)[:, 1]
    spend_prob = spend_model.predict_proba(x)[:, 1]
    pred_rev_if_spend = np.expm1(revenue_model.predict(x)).clip(min=0)
    expected_revenue = spend_prob * pred_rev_if_spend
    expected_clv = (1 - churn_prob) * expected_revenue
    expected_loss = churn_prob * expected_revenue

    out = df[["cutoff_date", "CustomerID"]].copy()
    out["churn_prob"] = churn_prob
    out["spend_prob"] = spend_prob
    out["pred_rev_if_spend"] = pred_rev_if_spend
    out["expected_revenue"] = expected_revenue
    out["expected_clv"] = expected_clv
    out["expected_loss"] = expected_loss

    if out["CustomerID"].isna().any():
        raise ValueError("Null CustomerID in predictions")
    if out.duplicated(subset=["cutoff_date", "CustomerID"]).any():
        raise ValueError("Duplicate keys in predictions")

    cutoffs = sorted(out["cutoff_date"].astype(str).unique())
    for cutoff in cutoffs:
        cdf = out[out["cutoff_date"].astype(str) == cutoff]
        table_name = f"{table_out_prefix}_{cutoff.replace('-', '_')}"
        con.register("pred_tmp", cdf)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM pred_tmp")

    latest = cutoffs[-1]
    latest_table = f"{table_out_prefix}_{latest.replace('-', '_')}"
    con.execute(f"CREATE OR REPLACE VIEW predictions_customer_latest AS SELECT * FROM {latest_table}")
    con.execute("CREATE OR REPLACE TABLE predictions_customer AS SELECT * FROM predictions_customer_latest")
    con.close()
    return latest_table
