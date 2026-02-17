import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Online Retail dataset.
    Returns cleaned dataframe ready for warehouse storage.
    """

    # Drop rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"]).copy()

    # Convert types
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Remove zero quantity or zero price
    df = df[(df["Quantity"] != 0) & (df["UnitPrice"] > 0)]

    # Create revenue columns
    df["line_revenue"] = df["Quantity"] * df["UnitPrice"]

    df["gross_revenue"] = df["line_revenue"].clip(lower=0)
    df["return_revenue"] = (-df["line_revenue"]).clip(lower=0)
    df["net_revenue"] = df["line_revenue"]

    # Flag cancelled invoices
    df["is_cancelled"] = df["InvoiceNo"].astype(str).str.startswith("C")

    return df
