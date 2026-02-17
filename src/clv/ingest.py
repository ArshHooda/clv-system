from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import get_connection

REQUIRED_COLUMNS = [
    "InvoiceDate",
    "CustomerID",
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "UnitPrice",
]


def ingest_transactions(csv_path: str, db_path: str = "data/warehouse.duckdb") -> int:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True)
    if df["InvoiceDate"].isna().any():
        raise ValueError("InvoiceDate contains unparsable values")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    if df["CustomerID"].isna().any():
        raise ValueError("CustomerID contains null/invalid values")

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0)
    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["revenue"] = df["Quantity"] * df["UnitPrice"]

    con = get_connection(db_path)
    con.execute("DROP TABLE IF EXISTS fact_transactions")
    con.register("staging_transactions", df)
    con.execute(
        """
        CREATE TABLE fact_transactions AS
        SELECT
            CAST(InvoiceDate AS TIMESTAMP) AS InvoiceDate,
            CAST(CustomerID AS BIGINT) AS CustomerID,
            CAST(InvoiceNo AS VARCHAR) AS InvoiceNo,
            CAST(StockCode AS VARCHAR) AS StockCode,
            CAST(Quantity AS DOUBLE) AS Quantity,
            CAST(UnitPrice AS DOUBLE) AS UnitPrice,
            CAST(revenue AS DOUBLE) AS revenue
        FROM staging_transactions
        """
    )
    inserted = con.execute("SELECT COUNT(*) FROM fact_transactions").fetchone()[0]
    con.close()
    return inserted
