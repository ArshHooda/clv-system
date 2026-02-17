import duckdb

DB_PATH = "data/warehouse.duckdb"

def main():
    con = duckdb.connect(DB_PATH)

    print("\n=== TABLES ===")
    print(con.execute("SHOW TABLES").fetchdf())

    print("\n=== predictions_customer (exists?) ===")
    try:
        print("Row count:", con.execute("SELECT COUNT(*) FROM predictions_customer").fetchone()[0])
        print("\nSample rows:")
        print(con.execute("SELECT * FROM predictions_customer LIMIT 10").fetchdf())
    except Exception as e:
        print("ERROR:", e)

    con.close()

if __name__ == "__main__":
    main()
