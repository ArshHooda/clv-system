from clv.db import get_connection
from clv.windows import compute_windows
import yaml
from clv.build_features import build_features
from clv.labels import build_labels_sql
from clv.rolling import build_rolling_dataset


def test_windows():
    from clv.db import get_connection
    from clv.windows import compute_windows
    import yaml

    con = get_connection()
    max_date = con.execute("SELECT MAX(InvoiceDate) FROM fact_transactions").fetchone()[0]
    con.close()

    with open("configs/params.yaml") as f:
        config = yaml.safe_load(f)
    
    if config.get("rolling", {}).get("enabled", False):
        build_rolling_dataset(config)
        return


    windows = compute_windows(
        max_date,
        config["data"]["observation_days"],
        config["data"]["gap_days"],
        config["data"]["prediction_days"],
    )

    print("\nComputed Windows:")
    for k, v in windows.items():
        print(f"{k}: {v}")

    build_features(windows)
    con = get_connection()
    label_sql = build_labels_sql(windows)
    con.execute(label_sql)

    print("Customer labels created.")

    churn_rate = con.execute(
        "SELECT AVG(churn_label) FROM customer_labels"
    ).fetchone()[0]

    print("Churn rate in snapshot:", round(churn_rate, 4))

    con.close()

    con = get_connection()

    con.execute("""
        CREATE OR REPLACE TABLE customer_model_data AS
        SELECT
            f.*,
            l.churn_label,
            l.revenue_pred_window
        FROM customer_features f
        JOIN customer_labels l
            ON f.CustomerID = l.CustomerID
    """)

    print("Model dataset created.")

    shape = con.execute(
        "SELECT COUNT(*) FROM customer_model_data"
    ).fetchone()[0]

    print("Final model rows:", shape)

    con.close()




if __name__ == "__main__":
    test_windows()
