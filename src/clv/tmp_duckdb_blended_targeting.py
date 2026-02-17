"""
tmp_duckdb_blended_targeting.py

Notebook-style quick test to:
1) Load predictions_customer (latest cutoff only)
2) Build a blended score using percentile ranks:
   blended = w_loss * rank(expected_loss) + w_clv * rank(expected_clv)
3) Optimize under BOTH constraints:
   - budget
   - max_customers
4) Print summary + top targets

Run:
    python src/clv/tmp_duckdb_blended_targeting.py
"""

import duckdb
import pandas as pd


DB_PATH = "data/warehouse.duckdb"


def add_percentile_rank(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    # Percentile rank in [0,1], higher is better
    df[out_col] = df[col].rank(pct=True, method="average")
    return df


def optimize_targeting(
    df: pd.DataFrame,
    budget_eur: float,
    cost_per_customer: float,
    max_customers: int,
    save_rate: float,
    score_col: str,
):
    df = df.copy().sort_values(score_col, ascending=False)

    # capacity constraint
    df = df.head(max_customers).copy()

    # budget constraint
    affordable_n = int(budget_eur // cost_per_customer)
    affordable_n = max(0, affordable_n)

    targeted = df.head(min(len(df), affordable_n)).copy()

    total_cost = len(targeted) * cost_per_customer

    # For prevented loss, we should still use expected_loss (not blended score)
    expected_prevented_loss = save_rate * targeted["expected_loss"].sum()

    net_uplift = expected_prevented_loss - total_cost
    roi = (net_uplift / total_cost) if total_cost > 0 else None

    return {
        "targeted_customers": int(len(targeted)),
        "total_cost": float(total_cost),
        "expected_prevented_loss": float(expected_prevented_loss),
        "net_uplift": float(net_uplift),
        "roi": None if roi is None else float(roi),
        "target_list": targeted,
    }


def main():
    # 1) Load
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
        FROM predictions_customer
    """).fetchdf()

    con.close()

    latest_cutoff = df["cutoff_date"].max()
    snap = df[df["cutoff_date"] == latest_cutoff].copy()

    # Safety (should already be unique now)
    snap = snap.drop_duplicates(subset=["cutoff_date", "CustomerID"])

    print("\n=== Latest snapshot ===")
    print("latest_cutoff:", latest_cutoff)
    print("rows:", len(snap), "| customers:", snap["CustomerID"].nunique())

    # 2) Build blended score (percentile ranks)
    w_loss = 0.7
    w_clv = 0.3

    snap = add_percentile_rank(snap, "expected_loss", "loss_rank")
    snap = add_percentile_rank(snap, "expected_clv", "clv_rank")

    snap["blended_score"] = (w_loss * snap["loss_rank"]) + (w_clv * snap["clv_rank"])

    # 3) Inputs (edit freely)
    budget_eur = 500.0
    cost_per_customer = 1.0
    max_customers = 2000
    save_rate = 0.15

    print("\n=== Inputs ===")
    print({
        "budget_eur": budget_eur,
        "cost_per_customer": cost_per_customer,
        "max_customers": max_customers,
        "save_rate": save_rate,
        "w_loss": w_loss,
        "w_clv": w_clv,
    })

    # 4) Optimize using blended score
    result = optimize_targeting(
        df=snap,
        budget_eur=budget_eur,
        cost_per_customer=cost_per_customer,
        max_customers=max_customers,
        save_rate=save_rate,
        score_col="blended_score",
    )

    print("\n=== Output summary ===")
    print({k: v for k, v in result.items() if k != "target_list"})

    targets = result["target_list"].copy()

    print("\n=== Top 10 targets (by blended_score) ===")
    show_cols = [
        "CustomerID",
        "blended_score",
        "expected_loss",
        "expected_clv",
        "expected_revenue",
        "churn_prob",
        "spend_prob",
        "loss_rank",
        "clv_rank",
    ]
    print(targets[show_cols].head(10).to_string(index=False))

    # Optional: compare what you'd get if you ranked by expected_loss only
    loss_only = snap.sort_values("expected_loss", ascending=False).head(len(targets))
    overlap = set(targets["CustomerID"]).intersection(set(loss_only["CustomerID"]))
    print("\nOverlap vs expected_loss-only selection:", round(len(overlap) / len(targets), 3))


if __name__ == "__main__":
    main()
