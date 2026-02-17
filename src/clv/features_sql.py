def build_customer_features_sql(windows):
    obs_start = windows["observation_start"]
    obs_end = windows["observation_end"]
    obs_end_date = windows["observation_end"].date()

    return f"""
    CREATE OR REPLACE TABLE customer_features AS
    WITH obs AS (
        SELECT *
        FROM fact_transactions
        WHERE InvoiceDate >= TIMESTAMP '{obs_start}'
          AND InvoiceDate <  TIMESTAMP '{obs_end}'
    ),
    base AS (
        SELECT
            CustomerID,

            COUNT(*) AS txn_count_obs,
            COUNT(DISTINCT InvoiceNo) AS invoice_count_obs,

            SUM(gross_revenue) AS gross_revenue_obs,
            SUM(return_revenue) AS return_revenue_obs,
            SUM(net_revenue) AS net_revenue_obs,

            AVG(net_revenue) AS avg_revenue_per_line_obs,

            MAX(InvoiceDate) AS last_purchase_date_obs,
            MIN(InvoiceDate) AS first_purchase_date_obs,

            DATEDIFF('day', MAX(InvoiceDate), DATE '{obs_end_date}') AS recency_days_obs,
            DATEDIFF('day', MIN(InvoiceDate), DATE '{obs_end_date}') AS tenure_days_obs,

            COUNT(DISTINCT CAST(InvoiceDate AS DATE)) AS active_days_obs

        FROM obs
        GROUP BY CustomerID
    ),
    recent AS (
        SELECT
            CustomerID,

            COUNT(DISTINCT CASE
                WHEN InvoiceDate >= TIMESTAMP '{obs_end}' - INTERVAL 30 DAY THEN InvoiceNo END
            ) AS invoice_count_30d,

            COUNT(DISTINCT CASE
                WHEN InvoiceDate >= TIMESTAMP '{obs_end}' - INTERVAL 90 DAY THEN InvoiceNo END
            ) AS invoice_count_90d,

            COUNT(CASE
                WHEN InvoiceDate >= TIMESTAMP '{obs_end}' - INTERVAL 30 DAY THEN 1 END
            ) AS txn_count_30d,

            COUNT(CASE
                WHEN InvoiceDate >= TIMESTAMP '{obs_end}' - INTERVAL 90 DAY THEN 1 END
            ) AS txn_count_90d,

            SUM(CASE
                WHEN InvoiceDate >= TIMESTAMP '{obs_end}' - INTERVAL 30 DAY THEN net_revenue ELSE 0 END
            ) AS net_revenue_30d,

            SUM(CASE
                WHEN InvoiceDate >= TIMESTAMP '{obs_end}' - INTERVAL 90 DAY THEN net_revenue ELSE 0 END
            ) AS net_revenue_90d

        FROM obs
        GROUP BY CustomerID
    ),

    cadence2 AS (
        SELECT
            CustomerID,
            AVG(DATEDIFF('day', prev_date, inv_date)) AS avg_days_between_invoices
        FROM (
            SELECT
                CustomerID,
                inv_date,
                LAG(inv_date) OVER (PARTITION BY CustomerID ORDER BY inv_date) AS prev_date
            FROM (
                SELECT CustomerID, CAST(MIN(InvoiceDate) AS DATE) AS inv_date
                FROM obs
                GROUP BY CustomerID, InvoiceNo
            )
        )
        WHERE prev_date IS NOT NULL
        GROUP BY CustomerID
    )
    SELECT
        b.CustomerID,

        b.txn_count_obs,
        b.invoice_count_obs,
        b.gross_revenue_obs,
        b.return_revenue_obs,
        b.net_revenue_obs,
        b.avg_revenue_per_line_obs,
        b.last_purchase_date_obs,
        b.first_purchase_date_obs,
        b.recency_days_obs,
        b.tenure_days_obs,
        b.active_days_obs,

        r.invoice_count_30d,
        r.invoice_count_90d,
        r.txn_count_30d,
        r.txn_count_90d,
        r.net_revenue_30d,
        r.net_revenue_90d,

        COALESCE(c2.avg_days_between_invoices, NULL) AS avg_days_between_invoices,

        CASE
            WHEN b.gross_revenue_obs > 0 THEN b.return_revenue_obs / b.gross_revenue_obs
            ELSE 0
        END AS return_ratio_obs

    FROM base b
    LEFT JOIN recent r ON b.CustomerID = r.CustomerID
    LEFT JOIN cadence2 c2 ON b.CustomerID = c2.CustomerID
    """
