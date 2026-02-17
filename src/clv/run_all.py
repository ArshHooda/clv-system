"""
run_all.py

One-command end-to-end runner:
- Builds rolling dataset (via pipeline.py config switch)
- Trains churn model
- Trains revenue hurdle models (spend + conditional revenue)
- Scores & writes predictions_customer to DuckDB

Run:
    python src/clv/run_all.py
"""

from clv.pipeline import test_windows
from clv.train_churn import train_churn_model
from clv.train_revenue import train_revenue_models
from clv.score import load_model, score_clv_and_write_to_db
from clv.run_report import main as run_report



def main():
    # 1) Build data tables (rolling enabled in configs/params.yaml triggers rolling build)
    test_windows()

    # 2) Train churn model (+ writes churn model artifact)
    train_churn_model()

    # 3) Train revenue models (+ writes spend/revenue/feature_cols artifacts)
    train_revenue_models()

    # 4) Score and write predictions store
    churn = load_model("artifacts/models/churn_xgb.joblib")
    spend = load_model("artifacts/models/spend_clf.joblib")
    rev = load_model("artifacts/models/revenue_reg.joblib")
    feature_cols = load_model("artifacts/models/feature_cols.joblib")

    score_clv_and_write_to_db(
        churn_model=churn,
        spend_model=spend,
        revenue_model=rev,
        feature_cols=feature_cols,
        table_in="customer_model_data_rollup",
        table_out_prefix="predictions_customer",
    )


    print("\n✅ run_all complete. DuckDB table ready: predictions_customer_latest")


if __name__ == "__main__":
    main()
    run_report()

