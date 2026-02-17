from src.clv.ingest import ingest_transactions
from src.clv.labels import build_labels_sql
from src.clv.features_sql import build_customer_features_sql
from src.clv.db import get_connection
from src.clv.windows import compute_windows_from_cutoff


def test_features_labels_create(test_config):
    ingest_transactions(test_config["data"]["input_csv"], test_config["data"]["db_path"])
    con = get_connection(test_config["data"]["db_path"])
    w = compute_windows_from_cutoff(con.execute("SELECT DATE '2010-06-01'").fetchone()[0], 60, 14, 30)
    con.execute(build_customer_features_sql(w))
    con.execute(build_labels_sql(w))
    assert con.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0] > 0
    assert con.execute("SELECT COUNT(*) FROM customer_labels WHERE churn_label IN (0,1)").fetchone()[0] > 0
    con.close()
