from __future__ import annotations

from src.clv.config import load_config
from src.clv.db import get_connection


def get_db():
    cfg = load_config()
    con = get_connection(cfg["data"]["db_path"])
    try:
        yield con
    finally:
        con.close()
