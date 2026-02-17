from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.deps import get_db
from src.api.schemas import PredictionsSummary

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/latest/summary", response_model=PredictionsSummary)
def latest_summary(con=Depends(get_db)):
    row = con.execute(
        "SELECT COUNT(*), AVG(churn_prob), AVG(expected_loss), AVG(expected_clv) FROM predictions_customer_latest"
    ).fetchone()
    return {"rows": row[0], "avg_churn_prob": row[1] or 0, "avg_expected_loss": row[2] or 0, "avg_expected_clv": row[3] or 0}


@router.get("/latest/top")
def latest_top(by: str = Query("expected_loss"), n: int = Query(50), con=Depends(get_db)):
    if by not in {"expected_loss", "expected_clv", "expected_revenue", "churn_prob"}:
        by = "expected_loss"
    return con.execute(f"SELECT * FROM predictions_customer_latest ORDER BY {by} DESC LIMIT ?", [n]).fetch_df().to_dict(orient="records")
