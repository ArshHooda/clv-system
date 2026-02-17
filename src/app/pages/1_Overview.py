from __future__ import annotations

import duckdb
import streamlit as st

st.title("Overview")
con = duckdb.connect("data/warehouse.duckdb")
try:
    df = con.execute("SELECT * FROM predictions_customer_latest").fetch_df()
    st.metric("Customers", len(df))
    st.metric("Avg churn", float(df["churn_prob"].mean()))
    st.metric("Avg expected loss", float(df["expected_loss"].mean()))
    st.bar_chart(df[["expected_loss", "expected_clv"]].head(50))
except Exception as e:
    st.warning(f"Run pipeline first: {e}")
finally:
    con.close()
