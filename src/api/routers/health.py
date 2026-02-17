from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_db
from src.api.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health(con=Depends(get_db)):
    latest = con.execute("SELECT MAX(cutoff_date) FROM predictions_customer_latest").fetchone()[0]
    rows = con.execute("SELECT COUNT(*) FROM predictions_customer_latest").fetchone()[0]
    return {"status": "ok", "latest_cutoff": str(latest) if latest else None, "rows_latest": rows}
