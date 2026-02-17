"""
run_report.py

Standardized run report artifact:
- Loads predictions_customer_latest (latest snapshot)
- Computes:
  1) Loss-only targeting (expected_loss)
  2) Blended targeting (rank-normalized expected_loss + expected_clv)
- Applies BOTH constraints:
  - budget
  - max_customers
- Saves:
  - JSON run summary to artifacts/reports/
  - CSV target lists (loss-only + blended)

Run:
    python src/clv/run_report.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = "data/warehouse.duckdb"
PRED_VIEW = "predictions_customer_latest"


@dataclass
class StrategySummary:
    targeted_customers: int
    total_cost: float
    expected_prevented_loss: float
    net_uplift: float
    roi: float | None


def add_percentile_rank(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    # percentile rank in [0,1]
    df[out_col] = df[col].rank(pct=True, method="average")
    return df


def optimize_targeting(
    df: pd.DataFrame,
    budget_eur: float,
    cost_per_customer: float,
    max_customers: int,
    save_rate: float,
    score_col: str,
) -> tuple[pd.DataFrame, StrategySummary]:
    df = df.copy().sort_values(score_col, ascending=False)

    # capacity constraint
    df = df.head(max_customers).copy()

    # budget constraint
    affordable_n = int(budget_eur // cost_per_customer) if cost_per_customer > 0 else 0
    affordable_n = max(0, affordable_n)

    targeted = df.head(min(len(df), affordable_n)).copy()

    total_cost = float(len(targeted) * cost_per_customer)

    # prevented loss must always be based on expected_loss (business truth)
    expected_prevented_loss = float(save_rate * targeted["expected_loss"].sum())
    net_uplift = float(expected_prevented_loss - total_cost)
    roi = (net_uplift / total_cost) if total_cost > 0 else None

    return targeted, StrategySummary(
        targeted_customers=int(len(targeted)),
        total_cost=total_cost,
        expected_prevented_loss=expected_prevented_loss,
        net_uplift=net_uplift,
        roi=None if roi is None else float(roi),
    )


def main():
    # -------------------------
    # 1) Load latest snapshot
    # -------------------------
    con = duckdb.connect(DB_PATH)
    df = con.execute(f"""
        SELECT
          cutoff_date,
          CustomerID,
          churn_prob,
          spend_prob,
          expected_revenue,
          expected_clv,
          expected_loss
        FROM {PRED_VIEW}
    """).fetchdf()
    con.close()

    if df.empty:
        raise ValueError(f"{PRED_VIEW} returned 0 rows. Run run_all.py first.")

    # Safety: one row per customer
    df = df.drop_duplicates(subset=["cutoff_date", "CustomerID"]).copy()

    latest_cutoff = pd.to_datetime(df["cutoff_date"]).max()
    cutoff_str = str(latest_cutoff.date())

    print("\n=== Snapshot ===")
    print("latest_cutoff:", latest_cutoff)
    print("rows:", len(df), "| customers:", df["CustomerID"].nunique())

    # -------------------------
    # 2) Assumptions / knobs
    # -------------------------
    budget_eur = 500.0
    cost_per_customer = 1.0
    max_customers = 2000
    save_rate = 0.15

    # Strategy weights (your chosen default)
    w_loss, w_clv = 0.7, 0.3
    top_n_preview = 20  # for report preview in JSON

    assumptions = {
        "budget_eur": budget_eur,
        "cost_per_customer": cost_per_customer,
        "max_customers": max_customers,
        "save_rate": save_rate,
        "w_loss": w_loss,
        "w_clv": w_clv,
        "source": PRED_VIEW,
        "db_path": DB_PATH,
    }

    # -------------------------
    # 3) Strategy A: loss-only
    # -------------------------
    targets_loss, summary_loss = optimize_targeting(
        df, budget_eur, cost_per_customer, max_customers, save_rate, score_col="expected_loss"
    )

    # -------------------------
    # 4) Strategy B: blended
    # -------------------------
    df2 = df.copy()
    df2 = add_percentile_rank(df2, "expected_loss", "loss_rank")
    df2 = add_percentile_rank(df2, "expected_clv", "clv_rank")
    df2["blended_score"] = w_loss * df2["loss_rank"] + w_clv * df2["clv_rank"]

    targets_blend, summary_blend = optimize_targeting(
        df2, budget_eur, cost_per_customer, max_customers, save_rate, score_col="blended_score"
    )

    # -------------------------
    # 5) Overlap
    # -------------------------
    overlap = set(targets_loss["CustomerID"]).intersection(set(targets_blend["CustomerID"]))
    overlap_pct = float(len(overlap) / max(1, len(targets_blend)))

    # -------------------------
    # 6) Save artifacts
    # -------------------------
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{cutoff_str}_{ts}"

    loss_csv = out_dir / f"top_loss_{base}.csv"
    blend_csv = out_dir / f"top_blended_{base}.csv"
    report_json = out_dir / f"run_report_{base}.json"

    # Keep the CSVs “analysis-ready”
    targets_loss.to_csv(loss_csv, index=False)
    targets_blend.to_csv(blend_csv, index=False)

    report = {
        "cutoff_date": cutoff_str,
        "generated_at": ts,
        "assumptions": assumptions,
        "strategy_loss_only": asdict(summary_loss),
        "strategy_blended": asdict(summary_blend),
        "overlap_pct_blended_vs_loss_only": overlap_pct,
        "files": {
            "loss_csv": str(loss_csv).replace("\\", "/"),
            "blended_csv": str(blend_csv).replace("\\", "/"),
            "report_json": str(report_json).replace("\\", "/"),
        },
        "top_preview": {
            "loss_only_top": targets_loss[["CustomerID", "expected_loss", "expected_clv", "churn_prob", "spend_prob"]]
                .head(top_n_preview)
                .to_dict(orient="records"),
            "blended_top": targets_blend[["CustomerID", "expected_loss", "expected_clv", "churn_prob", "spend_prob"]]
                .head(top_n_preview)
                .to_dict(orient="records"),
        },
    }

    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # -------------------------
    # 7) Console output
    # -------------------------
    print("\n=== Report saved ===")
    print("loss CSV   :", loss_csv)
    print("blended CSV:", blend_csv)
    print("JSON report:", report_json)

    print("\n=== Summary ===")
    print("Loss-only:", asdict(summary_loss))
    print("Blended  :", asdict(summary_blend))
    print("Overlap %:", round(overlap_pct, 3))


if __name__ == "__main__":
    main()
