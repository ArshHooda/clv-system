from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.title("Reports")
rdir = Path("artifacts/reports")
files = sorted([p for p in rdir.glob("*") if p.is_file()])
for f in files:
    st.write(f.name)
    if f.suffix == ".json":
        st.json(json.loads(f.read_text(encoding="utf-8")))
    with f.open("rb") as fp:
        st.download_button(f"Download {f.name}", fp.read(), file_name=f.name)
