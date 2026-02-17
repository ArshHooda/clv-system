from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .routers import decisioning, health, predictions

app = FastAPI(title="CLV Retention Decision Engine API")
app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(decisioning.router)


@app.get("/reports/list")
def list_reports():
    p = Path("artifacts/reports")
    p.mkdir(parents=True, exist_ok=True)
    return sorted([x.name for x in p.iterdir() if x.is_file()])


@app.get("/reports/{filename}")
def get_report(filename: str):
    fp = Path("artifacts/reports") / filename
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(fp)
