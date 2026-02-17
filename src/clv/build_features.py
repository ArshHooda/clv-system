from clv.db import get_connection
from clv.features_sql import build_customer_features_sql


def build_features(windows):
    con = get_connection()

    sql = build_customer_features_sql(windows)
    con.execute(sql)

    print("Customer features table created.")

    count = con.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0]
    print("Customers in snapshot:", count)

    con.close()
