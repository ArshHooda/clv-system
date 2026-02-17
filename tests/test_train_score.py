from src.clv.db import get_connection
from src.clv.ingest import ingest_transactions
from src.clv.rolling import build_rolling_dataset
from src.clv.score import score_clv_and_write_to_db
from src.clv.train_churn import train_churn_model
from src.clv.train_revenue import train_revenue_models


def test_train_and_score(test_config):
    ingest_transactions(test_config["data"]["input_csv"], test_config["data"]["db_path"])
    build_rolling_dataset(test_config)
    churn, cols, _ = train_churn_model(test_config)
    spend, rev, _ = train_revenue_models(test_config)
    score_clv_and_write_to_db(churn, spend, rev, cols, "customer_model_data_rollup", "predictions_customer", test_config["data"]["db_path"])
    con = get_connection(test_config["data"]["db_path"])
    assert con.execute("SELECT COUNT(*) FROM predictions_customer_latest").fetchone()[0] > 0
    con.close()
