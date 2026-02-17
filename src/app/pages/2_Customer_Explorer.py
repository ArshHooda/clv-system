from __future__ import annotations

import duckdb
import streamlit as st

st.title("Customer Explorer")
con = duckdb.connect("data/warehouse.duckdb")
try:
    df = con.execute("SELECT * FROM predictions_customer_latest").fetch_df()
    cid = st.number_input("CustomerID", min_value=int(df.CustomerID.min()), max_value=int(df.CustomerID.max()), value=int(df.CustomerID.min()))
    st.dataframe(df[df["CustomerID"] == cid])
except Exception as e:
    st.warning(f"Run pipeline first: {e}")
finally:
    con.close()
