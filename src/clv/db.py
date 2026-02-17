import duckdb
from pathlib import Path


DB_PATH = Path("data/warehouse.duckdb")


def get_connection():
    return duckdb.connect(DB_PATH)


def inspect_warehouse():
    con = get_connection()
    
    print("\nTables:")
    print(con.execute("SHOW TABLES").fetchdf())

    print("\nFact Table Sample:")
    print(con.execute("SELECT * FROM fact_transactions LIMIT 5").fetchdf())

    print("\nDate Range:")
    print(con.execute("SELECT MIN(InvoiceDate), MAX(InvoiceDate) FROM fact_transactions").fetchdf())

    print("\nCustomer Count:")
    print(con.execute("SELECT COUNT(DISTINCT CustomerID) FROM fact_transactions").fetchdf())

    con.close()


if __name__ == "__main__":
    inspect_warehouse()





    

