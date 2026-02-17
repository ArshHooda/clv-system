"""
tmp_decisioning_report.py

Notebook-style report:
- Loads predictions_customer
- Uses latest cutoff snapshot (real-world)
- Compares:
  1) Loss-only targeting (expected_loss)
  2) Blended targeting (rank-normalized loss + clv)
- Applies BOTH constraints:
  - budget
  - max_customers
- Prints summary + exports target list

Run:
    python src/clv/tmp_decisioning_report.py
"""

import duckdb
import pandas as pd
from pathlib import Path
from clv.reporting import save_run_artifacts


DB_PATH = "data/warehouse.duckdb"


def add_percentile_rank(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    df[out_col] = df[col].rank(pct=True, method="average")
    return df


def optimize_targeting(df, budget_eur, cost_per_customer, max_customers, save_rate, score_col):
    df = df.copy().sort_values(score_col, ascending=False)

    # capacity constraint
    df = df.head(max_customers).copy()

    # budget constraint
    affordable_n = int(budget_eur // cost_per_customer)
    affordable_n = max(0, affordable_n)

    targeted = df.head(min(len(df), affordable_n)).copy()

    total_cost = len(targeted) * cost_per_customer

    # prevented loss must be based on expected_loss, not blended score
    expected_prevented_loss = save_rate * targeted["expected_loss"].sum()
    net_uplift = expected_prevented_loss - total_cost
    roi = (net_uplift / total_cost) if total_cost > 0 else None

    return targeted, {
        "targeted_customers": int(len(targeted)),
        "total_cost": float(total_cost),
        "expected_prevented_loss": float(expected_prevented_loss),
        "net_uplift": float(net_uplift),
        "roi": None if roi is None else float(roi),
    }


def main():
    con = duckdb.connect(DB_PATH)
    df = con.execute("""
        SELECT
          cutoff_date,
          CustomerID,
          churn_prob,
          spend_prob,
          expected_revenue,
          expected_clv,
          expected_loss
        FROM predictions_customer_latest
    """).fetchdf()
    con.close()

    snap = df.copy()
    snap = snap.drop_duplicates(subset=["cutoff_date", "CustomerID"])  # safety
    latest_cutoff = pd.to_datetime(snap["cutoff_date"]).max()

    print("\n=== Snapshot ===")
    print("latest_cutoff:", latest_cutoff)
    print("rows:", len(snap), "| customers:", snap["CustomerID"].nunique())

    # ---- assumptions ----
    budget_eur = 500.0
    cost_per_customer = 1.0
    max_customers = 2000
    save_rate = 0.15

    print("\n=== Assumptions ===")
    print({
        "budget_eur": budget_eur,
        "cost_per_customer": cost_per_customer,
        "max_customers": max_customers,
        "save_rate": save_rate
    })

    # ---- strategy 1: loss-only ----
    targets_loss, summary_loss = optimize_targeting(
        snap, budget_eur, cost_per_customer, max_customers, save_rate, score_col="expected_loss"
    )

    # ---- strategy 2: blended ----
    w_loss, w_clv = 0.7, 0.3

    snap2 = snap.copy()
    snap2 = add_percentile_rank(snap2, "expected_loss", "loss_rank")
    snap2 = add_percentile_rank(snap2, "expected_clv", "clv_rank")
    snap2["blended_score"] = w_loss * snap2["loss_rank"] + w_clv * snap2["clv_rank"]

    targets_blend, summary_blend = optimize_targeting(
        snap2, budget_eur, cost_per_customer, max_customers, save_rate, score_col="blended_score"
    )

    # ---- overlap ----
    overlap = set(targets_loss["CustomerID"]).intersection(set(targets_blend["CustomerID"]))
    overlap_pct = len(overlap) / max(1, len(targets_blend))

    print("\n=== Strategy comparison ===")
    print("Loss-only summary:", summary_loss)
    print("Blended summary  :", summary_blend)
    print("Overlap % (blend vs loss-only):", round(overlap_pct, 3))

    # ---- top targets ----
    print("\n=== Top 10 (loss-only) ===")
    print(targets_loss[["CustomerID", "expected_loss", "expected_clv", "churn_prob"]].head(10).to_string(index=False))

    print("\n=== Top 10 (blended) ===")
    show_cols = ["CustomerID", "blended_score", "expected_loss", "expected_clv", "churn_prob", "spend_prob"]
    print(targets_blend[show_cols].head(10).to_string(index=False))

    # ---- standardized artifacts (JSON + two CSVs) ----
    artifacts = save_run_artifacts(
        latest_cutoff=latest_cutoff,
        assumptions={
            "budget_eur": budget_eur,
            "cost_per_customer": cost_per_customer,
            "max_customers": max_customers,
            "save_rate": save_rate,
            "w_loss": w_loss,
            "w_clv": w_clv,
            "source": "predictions_customer_latest",
            "db_path": DB_PATH,
        },
        summary_loss=summary_loss,
        summary_blend=summary_blend,
        overlap_pct=overlap_pct,
        targets_loss=targets_loss,
        targets_blend=targets_blend,
        top_n_preview=20,
        out_dir="artifacts/reports",
    )

    print("\n✅ Saved standardized run artifacts:")
    print(" - Loss CSV   :", artifacts["loss_csv"])
    print(" - Blended CSV:", artifacts["blended_csv"])
    print(" - JSON report:", artifacts["report_json"])



if __name__ == "__main__":
    main()
