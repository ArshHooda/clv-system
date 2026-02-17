"""
tmp_duckdb_quick_test.py

Notebook-style quick test (no dashboard) to:
1) Read predictions_customer from DuckDB
2) Use ONLY the latest cutoff snapshot (real-world targeting)
3) Run budget + capacity constrained targeting (Strategy C)
4) Print summary + top targets

Run from project root:
    python src/clv/tmp_duckdb_quick_test.py
"""

import duckdb
import pandas as pd


DB_PATH = "data/warehouse.duckdb"


def optimize_targeting(
    df: pd.DataFrame,
    budget_eur: float,
    cost_per_customer: float,
    max_customers: int,
    save_rate: float,
    score_col: str = "expected_loss",
):
    # rank by score, then apply capacity + budget constraints
    df = df.copy().sort_values(score_col, ascending=False)

    # capacity constraint
    df = df.head(max_customers).copy()

    # budget constraint
    affordable_n = int(budget_eur // cost_per_customer)
    affordable_n = max(0, affordable_n)

    targeted = df.head(min(len(df), affordable_n)).copy()

    total_cost = len(targeted) * cost_per_customer
    expected_prevented = save_rate * targeted[score_col].sum()
    net_uplift = expected_prevented - total_cost
    roi = (net_uplift / total_cost) if total_cost > 0 else None

    return {
        "targeted_customers": int(len(targeted)),
        "total_cost": float(total_cost),
        "expected_prevented_loss": float(expected_prevented),
        "net_uplift": float(net_uplift),
        "roi": None if roi is None else float(roi),
        "target_list": targeted,
    }


def main():
    # ---- 1) Load from DuckDB ----
    con = duckdb.connect(DB_PATH)

    # sanity: show tables
    print("\n=== TABLES ===")
    print(con.execute("SHOW TABLES").fetchdf())

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

    print("\n=== DATASET ===")
    print("rows:", len(df))
    print("distinct cutoffs:", df["cutoff_date"].nunique())
    latest_cutoff = df["cutoff_date"].max()
    print("latest cutoff:", latest_cutoff)

    # ---- 2) Latest snapshot only (real-world ops) ----
    snap = df[df["cutoff_date"] == latest_cutoff].copy()
    print("customers in latest snapshot:", snap["CustomerID"].nunique())

    # ---- 3) Inputs (edit these freely) ----
    budget_eur = 500.0
    cost_per_customer = 1.0
    max_customers = 2000
    save_rate = 0.15
    score_col = "expected_loss"  # best retention targeting metric

    print("\n=== INPUTS ===")
    print({
        "budget_eur": budget_eur,
        "cost_per_customer": cost_per_customer,
        "max_customers": max_customers,
        "save_rate": save_rate,
        "score_col": score_col,
    })

    # ---- 4) Optimize ----
    result = optimize_targeting(
        df=snap,
        budget_eur=budget_eur,
        cost_per_customer=cost_per_customer,
        max_customers=max_customers,
        save_rate=save_rate,
        score_col=score_col,
    )

    print("\n=== OUTPUT SUMMARY ===")
    print({k: v for k, v in result.items() if k != "target_list"})

    targets = result["target_list"]

    print("\n=== TOP 10 TARGETS (by expected_loss) ===")
    show_cols = ["CustomerID", "expected_loss", "expected_clv", "expected_revenue", "churn_prob", "spend_prob"]
    print(targets[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
