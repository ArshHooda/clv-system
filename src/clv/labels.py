def build_labels_sql(windows):
    return f"""
    CREATE OR REPLACE TABLE customer_labels AS
    SELECT
        f.CustomerID,

        COALESCE(SUM(t.net_revenue), 0) AS revenue_pred_window,

        CASE
            WHEN COUNT(t.InvoiceNo) > 0 THEN 0
            ELSE 1
        END AS churn_label

    FROM customer_features f

    LEFT JOIN fact_transactions t
        ON f.CustomerID = t.CustomerID
        AND t.InvoiceDate >= TIMESTAMP '{windows["prediction_start"]}'
        AND t.InvoiceDate < TIMESTAMP '{windows["prediction_end"]}'

    GROUP BY f.CustomerID
    """
