from __future__ import annotations

import duckdb
import streamlit as st

from src.clv.decisioning import build_blended_score, optimize_targeting

st.title("Targeting Simulator")
con = duckdb.connect("data/warehouse.duckdb")
try:
    df = con.execute("SELECT * FROM predictions_customer_latest").fetch_df()
    budget = st.slider("Budget", 50, 5000, 500)
    cost = st.slider("Cost per customer", 1, 20, 1)
    max_customers = st.slider("Max customers", 10, 5000, 500)
    save_rate = st.slider("Save rate", 0.01, 1.0, 0.15)
    w_loss = st.slider("w_loss", 0.0, 1.0, 0.7)
    blended = build_blended_score(df, w_loss, 1 - w_loss)
    top, summary = optimize_targeting(blended, budget, cost, max_customers, save_rate, "blended_score")
    st.json(summary)
    st.dataframe(top.head(25))
except Exception as e:
    st.warning(f"Run pipeline first: {e}")
finally:
    con.close()
