import pandas as pd
from pathlib import Path

from clv.clean import clean_transactions
from clv.db import get_connection

DATA_PATH = Path("data/external/online_retail.xlsx")


def ingest():
    print("Reading raw data...")
    df = pd.read_excel(DATA_PATH)

    print("Cleaning data...")
    df_clean = clean_transactions(df)

    print("Rows after cleaning:", len(df_clean))

    con = get_connection()

    print("Writing to DuckDB...")
    con.execute("DROP TABLE IF EXISTS fact_transactions")
    con.register("df_clean", df_clean)

    con.execute("""
        CREATE TABLE fact_transactions AS
        SELECT * FROM df_clean
    """)

    con.close()

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
