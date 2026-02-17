from __future__ import annotations

import duckdb
import numpy as np
import streamlit as st

from src.clv.decisioning import weight_sweep

st.title("Weight Sweep")
con = duckdb.connect("data/warehouse.duckdb")
try:
    df = con.execute("SELECT * FROM predictions_customer_latest").fetch_df()
    res = weight_sweep(df, list(np.linspace(0, 1, 11)), 500, 1, 1000, 0.15)
    st.dataframe(res)
    st.line_chart(res.set_index("w_loss")["roi"])
except Exception as e:
    st.warning(f"Run pipeline first: {e}")
finally:
    con.close()
