from __future__ import annotations


def build_labels_sql(w: dict) -> str:
    return f"""
    CREATE OR REPLACE TABLE customer_labels AS
    WITH obs_customers AS (
        SELECT DISTINCT CustomerID
        FROM fact_transactions
        WHERE InvoiceDate >= TIMESTAMP '{w["obs_start"]}'
          AND InvoiceDate < TIMESTAMP '{w["obs_end"]}'
    ),
    pred AS (
        SELECT
            CustomerID,
            SUM(revenue)::DOUBLE AS revenue_pred_window,
            SUM(CASE WHEN revenue > 0 THEN 1 ELSE 0 END) AS purchases
        FROM fact_transactions
        WHERE InvoiceDate >= TIMESTAMP '{w["pred_start"]}'
          AND InvoiceDate < TIMESTAMP '{w["pred_end"]}'
        GROUP BY 1
    )
    SELECT
        o.CustomerID,
        CASE WHEN COALESCE(p.purchases, 0) = 0 THEN 1 ELSE 0 END AS churn_label,
        COALESCE(p.revenue_pred_window, 0.0) AS revenue_pred_window
    FROM obs_customers o
    LEFT JOIN pred p USING (CustomerID)
    """
