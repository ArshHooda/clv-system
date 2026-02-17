from clv.score import load_model, score_clv_and_write_to_db

def main():
    churn = load_model("artifacts/models/churn_xgb.joblib")
    spend = load_model("artifacts/models/spend_clf.joblib")
    rev   = load_model("artifacts/models/revenue_reg.joblib")
    feature_cols = load_model("artifacts/models/feature_cols.joblib")

    score_clv_and_write_to_db(
        churn_model=churn,
        spend_model=spend,
        revenue_model=rev,
        feature_cols=feature_cols,
        table_in="customer_model_data_rollup",
        table_out="predictions_customer"
    )

if __name__ == "__main__":
    main()
