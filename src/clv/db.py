from __future__ import annotations

from pathlib import Path

import duckdb

from .config import ROOT


def get_connection(db_path: str = "data/warehouse.duckdb") -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))
