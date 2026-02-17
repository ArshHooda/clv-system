from __future__ import annotations

from .db import get_connection
from .features_sql import build_customer_features_sql
from .labels import build_labels_sql
from .windows import compute_windows_from_cutoff, generate_cutoffs


def build_rolling_dataset(config: dict) -> dict:
    con = get_connection(config["data"]["db_path"])
    min_date, max_date = con.execute(
        "SELECT MIN(CAST(InvoiceDate AS DATE)), MAX(CAST(InvoiceDate AS DATE)) FROM fact_transactions"
    ).fetchone()

    cutoffs = generate_cutoffs(
        min_date,
        max_date,
        config["data"]["observation_days"],
        config["data"]["gap_days"],
        config["data"]["prediction_days"],
        config["rolling"]["step_days"],
    )

    con.execute("DROP TABLE IF EXISTS customer_model_data_rollup")
    con.execute("CREATE TABLE customer_model_data_rollup AS SELECT 1 as x WHERE 1=0")

    for i, cutoff in enumerate(cutoffs):
        w = compute_windows_from_cutoff(
            cutoff,
            config["data"]["observation_days"],
            config["data"]["gap_days"],
            config["data"]["prediction_days"],
        )
        con.execute(build_customer_features_sql(w))
        con.execute(build_labels_sql(w))
        con.execute(
            f"""
            CREATE OR REPLACE TABLE customer_model_data_tmp AS
            SELECT
                DATE '{cutoff}' AS cutoff_date,
                f.*,
                l.churn_label,
                l.revenue_pred_window
            FROM customer_features f
            JOIN customer_labels l USING (CustomerID)
            """
        )
        if i == 0:
            con.execute("CREATE OR REPLACE TABLE customer_model_data_rollup AS SELECT * FROM customer_model_data_tmp")
        else:
            con.execute("INSERT INTO customer_model_data_rollup SELECT * FROM customer_model_data_tmp")

    con.execute(
        """
        CREATE OR REPLACE TABLE customer_model_data_rollup AS
        SELECT DISTINCT * FROM customer_model_data_rollup
        """
    )
    dupes = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT cutoff_date, CustomerID, COUNT(*) c
            FROM customer_model_data_rollup
            GROUP BY 1,2
            HAVING c > 1
        )
        """
    ).fetchone()[0]
    if dupes > 0:
        raise ValueError("Duplicate (cutoff_date, CustomerID) in rollup")

    rows, distinct_cutoffs = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT cutoff_date) FROM customer_model_data_rollup"
    ).fetchone()
    latest_customers = con.execute(
        """
        SELECT COUNT(*) FROM customer_model_data_rollup
        WHERE cutoff_date = (SELECT MAX(cutoff_date) FROM customer_model_data_rollup)
        """
    ).fetchone()[0]
    con.close()
    return {"rows": rows, "distinct_cutoffs": distinct_cutoffs, "latest_customers": latest_customers}
