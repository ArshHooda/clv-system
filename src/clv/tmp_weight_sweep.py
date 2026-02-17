"""
tmp_weight_sweep.py

Notebook-style sweep for blended weights:
- For each (w_loss, w_clv), compute:
  - ROI (based on expected_loss prevented)
  - avg expected_clv of targeted customers
  - overlap vs loss-only
- Prints a results table sorted by ROI or avg CLV

Run:
    python src/clv/tmp_weight_sweep.py
"""

import duckdb
import pandas as pd

DB_PATH = "data/warehouse.duckdb"


def add_percentile_rank(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    df[out_col] = df[col].rank(pct=True, method="average")
    return df


def select_targets(df, budget_eur, cost_per_customer, max_customers, score_col):
    df = df.copy().sort_values(score_col, ascending=False)
    df = df.head(max_customers).copy()
    affordable_n = int(budget_eur // cost_per_customer)
    affordable_n = max(0, affordable_n)
    return df.head(min(len(df), affordable_n)).copy()


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

    latest_cutoff = df["cutoff_date"].max()
    snap = df[df["cutoff_date"] == latest_cutoff].copy()
    snap = snap.drop_duplicates(subset=["cutoff_date", "CustomerID"])

    # Assumptions
    budget_eur = 500.0
    cost_per_customer = 1.0
    max_customers = 2000
    save_rate = 0.15

    # Loss-only baseline
    base_targets = select_targets(snap, budget_eur, cost_per_customer, max_customers, "expected_loss")
    base_set = set(base_targets["CustomerID"])

    # Precompute ranks once
    snap = add_percentile_rank(snap, "expected_loss", "loss_rank")
    snap = add_percentile_rank(snap, "expected_clv", "clv_rank")

    weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
    rows = []

    for w_loss in weights:
        w_clv = 1.0 - w_loss
        snap["blended_score"] = w_loss * snap["loss_rank"] + w_clv * snap["clv_rank"]

        targets = select_targets(snap, budget_eur, cost_per_customer, max_customers, "blended_score")

        total_cost = len(targets) * cost_per_customer
        expected_prevented_loss = save_rate * targets["expected_loss"].sum()
        net_uplift = expected_prevented_loss - total_cost
        roi = (net_uplift / total_cost) if total_cost > 0 else None

        overlap = len(set(targets["CustomerID"]).intersection(base_set)) / max(1, len(targets))

        rows.append({
            "w_loss": w_loss,
            "w_clv": w_clv,
            "targeted": len(targets),
            "expected_prevented_loss": expected_prevented_loss,
            "roi": roi,
            "avg_expected_clv_targeted": float(targets["expected_clv"].mean()),
            "avg_expected_loss_targeted": float(targets["expected_loss"].mean()),
            "overlap_vs_loss_only": overlap,
        })

    out = pd.DataFrame(rows).sort_values(["roi"], ascending=False)
    print("\n=== Weight sweep results (sorted by ROI) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
