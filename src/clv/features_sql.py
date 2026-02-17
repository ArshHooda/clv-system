from __future__ import annotations


def build_customer_features_sql(w: dict) -> str:
    return f"""
    CREATE OR REPLACE TABLE customer_features AS
    WITH obs AS (
        SELECT *
        FROM fact_transactions
        WHERE InvoiceDate >= TIMESTAMP '{w["obs_start"]}'
          AND InvoiceDate < TIMESTAMP '{w["obs_end"]}'
    ),
    base AS (
        SELECT
            CustomerID,
            COUNT(*)::DOUBLE AS txn_count_obs,
            COUNT(DISTINCT InvoiceNo)::DOUBLE AS invoice_count_obs,
            SUM(revenue)::DOUBLE AS net_revenue_obs,
            SUM(CASE WHEN revenue > 0 THEN revenue ELSE 0 END)::DOUBLE AS gross_revenue_obs,
            SUM(CASE WHEN revenue < 0 THEN -revenue ELSE 0 END)::DOUBLE AS return_revenue_obs,
            AVG(revenue)::DOUBLE AS avg_revenue_per_line_obs,
            DATE_DIFF('day', MAX(CAST(InvoiceDate AS DATE)), DATE '{w["obs_end"]}')::DOUBLE AS recency_days_obs,
            DATE_DIFF('day', MIN(CAST(InvoiceDate AS DATE)), DATE '{w["obs_end"]}')::DOUBLE AS tenure_days_obs,
            COUNT(DISTINCT CAST(InvoiceDate AS DATE))::DOUBLE AS active_days_obs,
            COUNT(DISTINCT CASE WHEN InvoiceDate >= TIMESTAMP '{w["obs_end"]}' - INTERVAL 30 DAY THEN InvoiceNo END)::DOUBLE AS invoice_count_30d,
            COUNT(DISTINCT CASE WHEN InvoiceDate >= TIMESTAMP '{w["obs_end"]}' - INTERVAL 90 DAY THEN InvoiceNo END)::DOUBLE AS invoice_count_90d,
            SUM(CASE WHEN InvoiceDate >= TIMESTAMP '{w["obs_end"]}' - INTERVAL 30 DAY THEN 1 ELSE 0 END)::DOUBLE AS txn_count_30d,
            SUM(CASE WHEN InvoiceDate >= TIMESTAMP '{w["obs_end"]}' - INTERVAL 90 DAY THEN 1 ELSE 0 END)::DOUBLE AS txn_count_90d,
            SUM(CASE WHEN InvoiceDate >= TIMESTAMP '{w["obs_end"]}' - INTERVAL 30 DAY THEN revenue ELSE 0 END)::DOUBLE AS net_revenue_30d,
            SUM(CASE WHEN InvoiceDate >= TIMESTAMP '{w["obs_end"]}' - INTERVAL 90 DAY THEN revenue ELSE 0 END)::DOUBLE AS net_revenue_90d
        FROM obs
        GROUP BY 1
    ),
    inv AS (
        SELECT CustomerID, CAST(InvoiceDate AS DATE) AS invoice_day
        FROM obs
        GROUP BY 1,2
    ),
    inv_gap AS (
        SELECT
            CustomerID,
            AVG(
                DATE_DIFF('day', LAG(invoice_day) OVER (PARTITION BY CustomerID ORDER BY invoice_day), invoice_day)
            )::DOUBLE AS avg_days_between_invoices
        FROM inv
        QUALIFY LAG(invoice_day) OVER (PARTITION BY CustomerID ORDER BY invoice_day) IS NOT NULL
    )
    SELECT
        b.*,
        COALESCE(g.avg_days_between_invoices, NULL) AS avg_days_between_invoices,
        CASE WHEN b.gross_revenue_obs = 0 THEN NULL ELSE b.return_revenue_obs / NULLIF(b.gross_revenue_obs, 0) END AS return_ratio_obs
    FROM base b
    LEFT JOIN inv_gap g USING (CustomerID)
    """
