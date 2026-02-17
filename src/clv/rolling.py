from clv.db import get_connection
from clv.windows import generate_cutoffs, compute_windows_from_cutoff
from clv.features_sql import build_customer_features_sql
from clv.labels import build_labels_sql

def build_rolling_dataset(config):
    con = get_connection()

    min_date = con.execute("SELECT MIN(InvoiceDate) FROM fact_transactions").fetchone()[0]
    max_date = con.execute("SELECT MAX(InvoiceDate) FROM fact_transactions").fetchone()[0]

    obs_days = config["data"]["observation_days"]
    gap_days = config["data"]["gap_days"]
    pred_days = config["data"]["prediction_days"]
    step_days = config["rolling"]["step_days"]

    cutoffs = generate_cutoffs(min_date, max_date, obs_days, gap_days, pred_days, step_days)
    print(f"Generated cutoffs: {len(cutoffs)}")

    if len(cutoffs) == 0:
        raise ValueError("No valid cutoffs available. Reduce observation_days or prediction_days.")

    # Output table
    con.execute("DROP TABLE IF EXISTS customer_model_data_rollup")

    for cutoff in cutoffs:
        windows = compute_windows_from_cutoff(cutoff, obs_days, gap_days, pred_days)

        # Build temp feature/label tables for this cutoff
        con.execute(build_customer_features_sql(windows))
        con.execute(build_labels_sql(windows))

        # Merge to temp model table
        con.execute("""
            CREATE OR REPLACE TABLE customer_model_data_tmp AS
            SELECT
                CAST(? AS DATE) AS cutoff_date,
                f.*,
                l.churn_label,
                l.revenue_pred_window
            FROM customer_features f
            JOIN customer_labels l USING (CustomerID)
        """, [cutoff.date()])

        # Create rollup schema once (first loop), then append once per cutoff
        con.execute("""
            CREATE TABLE IF NOT EXISTS customer_model_data_rollup AS
            SELECT * FROM customer_model_data_tmp WHERE 1=0
        """)

        con.execute("""
            INSERT INTO customer_model_data_rollup
            SELECT * FROM customer_model_data_tmp
        """)

    # Quick stats
    rows = con.execute("SELECT COUNT(*) FROM customer_model_data_rollup").fetchone()[0]
    cut_count = con.execute("SELECT COUNT(DISTINCT cutoff_date) FROM customer_model_data_rollup").fetchone()[0]
    print("Rolling dataset rows:", rows)
    print("Distinct cutoffs:", cut_count)

    con.close()
