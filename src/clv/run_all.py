from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ROOT, load_config
from .db import get_connection
from .decisioning import build_blended_score, optimize_targeting
from .ingest import ingest_transactions
from .quality import calibration_check, data_drift_report, threshold_sanity, validate_prediction_store
from .reporting import save_run_artifacts
from .rolling import build_rolling_dataset
from .score import score_clv_and_write_to_db
from .train_churn import train_churn_model
from .train_revenue import train_revenue_models


def main():
    cfg = load_config()
    con = get_connection(cfg["data"]["db_path"])
    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='fact_transactions'"
    ).fetchone()[0]
    con.close()

    if not exists:
        ingest_transactions(str(ROOT / cfg["data"]["input_csv"]), cfg["data"]["db_path"])

    build_rolling_dataset(cfg)
    churn_model, feature_cols, churn_metrics = train_churn_model(cfg)
    spend_model, revenue_model, revenue_metrics = train_revenue_models(cfg)
    latest_table = score_clv_and_write_to_db(
        churn_model,
        spend_model,
        revenue_model,
        feature_cols,
        "customer_model_data_rollup",
        "predictions_customer",
        cfg["data"]["db_path"],
    )

    con = get_connection(cfg["data"]["db_path"])
    pred = con.execute("SELECT * FROM predictions_customer_latest").fetch_df()
    latest_cutoff = pred["cutoff_date"].max()

    dcfg = cfg["decisioning"]
    top_loss, loss_summary = optimize_targeting(
        pred,
        dcfg["default_budget_eur"],
        dcfg["default_cost_per_customer"],
        dcfg["default_max_customers"],
        dcfg["default_save_rate"],
        "expected_loss",
    )
    blended = build_blended_score(pred, dcfg["default_w_loss"], dcfg["default_w_clv"])
    top_blend, blend_summary = optimize_targeting(
        blended,
        dcfg["default_budget_eur"],
        dcfg["default_cost_per_customer"],
        dcfg["default_max_customers"],
        dcfg["default_save_rate"],
        "blended_score",
    )
    overlap = len(set(top_loss.CustomerID).intersection(set(top_blend.CustomerID))) / max(1, len(top_loss))

    save_run_artifacts(
        latest_cutoff,
        dcfg,
        loss_summary,
        blend_summary,
        overlap,
        top_loss,
        top_blend,
        ROOT / "artifacts/reports",
    )

    validate_prediction_store(con, "predictions_customer_latest")
    train_df = con.execute("SELECT * FROM customer_model_data_rollup").fetch_df()
    quality = {
        "churn_metrics": churn_metrics,
        "revenue_metrics": revenue_metrics,
        "drift": data_drift_report(train_df, pred),
        "calibration": calibration_check(
            (train_df["churn_label"] if "churn_label" in train_df.columns else pd.Series([0])).head(len(pred)),
            pred["churn_prob"],
        ),
        "thresholds": threshold_sanity(pred["churn_prob"]),
        "latest_prediction_table": latest_table,
    }
    with (Path(ROOT) / "artifacts/reports/quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)
    con.close()


if __name__ == "__main__":
    main()
