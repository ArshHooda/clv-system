from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return v


def save_run_artifacts(
    latest_cutoff,
    assumptions,
    loss_summary,
    blend_summary,
    overlap_pct,
    top_loss_df,
    top_blend_df,
    out_dir,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cutoff_txt = str(latest_cutoff).replace("-", "_")

    loss_csv = out / f"top_loss_{cutoff_txt}_{ts}.csv"
    blend_csv = out / f"top_blended_{cutoff_txt}_{ts}.csv"
    report_json = out / f"run_report_{cutoff_txt}_{ts}.json"

    top_loss_df.to_csv(loss_csv, index=False)
    top_blend_df.to_csv(blend_csv, index=False)

    payload = {
        "cutoff_date": str(latest_cutoff),
        "assumptions": {k: _safe(v) for k, v in assumptions.items()},
        "loss_summary": {k: _safe(v) for k, v in loss_summary.items()},
        "blend_summary": {k: _safe(v) for k, v in blend_summary.items()},
        "overlap_pct": float(overlap_pct),
        "top10_loss_ids": [int(x) for x in top_loss_df["CustomerID"].head(10).tolist()],
        "top10_blend_ids": [int(x) for x in top_blend_df["CustomerID"].head(10).tolist()],
        "dataset_stats": {
            "rows_latest": int(len(top_loss_df)),
            "generated_at": datetime.utcnow().isoformat(),
        },
    }
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return {"loss_csv": str(loss_csv), "blend_csv": str(blend_csv), "report_json": str(report_json)}
