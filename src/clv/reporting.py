from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

def _json_sanitize(obj):
    """
    Convert pandas/numpy objects to JSON-safe Python types.
    - Timestamp/date -> ISO string
    - numpy scalars -> Python scalars
    - NaN/NaT -> None
    """
    # pandas Timestamp / datetime-like
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    # numpy scalar types (np.float64, np.int64, etc.)
    if isinstance(obj, (np.generic,)):
        val = obj.item()
        # handle nan
        if isinstance(val, float) and (pd.isna(val)):
            return None
        return val

    # pandas NaT / NaN
    if obj is pd.NaT or (isinstance(obj, float) and pd.isna(obj)):
        return None

    # dict
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]

    # fallback: normal JSON-safe types pass through
    return obj



@dataclass
class StrategySummary:
    targeted_customers: int
    total_cost: float
    expected_prevented_loss: float
    net_uplift: float
    roi: float | None


def save_run_artifacts(
    *,
    latest_cutoff: pd.Timestamp,
    assumptions: dict,
    summary_loss: dict | StrategySummary,
    summary_blend: dict | StrategySummary,
    overlap_pct: float,
    targets_loss: pd.DataFrame,
    targets_blend: pd.DataFrame,
    top_n_preview: int = 20,
    out_dir: str = "artifacts/reports",
) -> dict:
    """
    Saves:
      - run_report_<cutoff>_<timestamp>.json
      - top_loss_<cutoff>_<timestamp>.csv
      - top_blended_<cutoff>_<timestamp>.csv

    Returns dict with file paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cutoff_str = str(pd.to_datetime(latest_cutoff).date())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{cutoff_str}_{ts}"

    loss_csv = out_path / f"top_loss_{base}.csv"
    blend_csv = out_path / f"top_blended_{base}.csv"
    report_json = out_path / f"run_report_{base}.json"

    targets_loss.to_csv(loss_csv, index=False)
    targets_blend.to_csv(blend_csv, index=False)

    def _to_dict(x):
        if hasattr(x, "__dataclass_fields__"):
            return asdict(x)
        return dict(x)

    report = {
        "cutoff_date": cutoff_str,
        "generated_at": ts,
        "assumptions": assumptions,
        "strategy_loss_only": _to_dict(summary_loss),
        "strategy_blended": _to_dict(summary_blend),
        "overlap_pct_blended_vs_loss_only": float(overlap_pct),
        "files": {
            "loss_csv": str(loss_csv).replace("\\", "/"),
            "blended_csv": str(blend_csv).replace("\\", "/"),
            "report_json": str(report_json).replace("\\", "/"),
        },
        "top_preview": {
            "loss_only_top": targets_loss.head(top_n_preview).to_dict(orient="records"),
            "blended_top": targets_blend.head(top_n_preview).to_dict(orient="records"),
        },
    }

    report_json.write_text(json.dumps(_json_sanitize(report), indent=2), encoding="utf-8")


    return {
        "loss_csv": loss_csv,
        "blended_csv": blend_csv,
        "report_json": report_json,
    }
