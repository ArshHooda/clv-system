from __future__ import annotations

import pandas as pd


def optimize_targeting(df, budget_eur, cost_per_customer, max_customers, save_rate, score_col):
    cap = min(int(budget_eur // cost_per_customer), int(max_customers))
    ranked = df.sort_values(score_col, ascending=False).head(cap).copy()
    ranked["prevented_loss"] = ranked["expected_loss"] * save_rate
    total_cost = len(ranked) * cost_per_customer
    prevented = float(ranked["prevented_loss"].sum())
    net_uplift = prevented - total_cost
    summary = {
        "targeted_customers": int(len(ranked)),
        "total_cost": float(total_cost),
        "expected_prevented_loss": prevented,
        "net_uplift": float(net_uplift),
        "roi": float(net_uplift / total_cost) if total_cost else 0.0,
    }
    return ranked, summary


def build_blended_score(df: pd.DataFrame, w_loss: float, w_clv: float) -> pd.DataFrame:
    out = df.copy()
    out["r_loss"] = out["expected_loss"].rank(pct=True)
    out["r_clv"] = out["expected_clv"].rank(pct=True)
    out["blended_score"] = w_loss * out["r_loss"] + w_clv * out["r_clv"]
    return out


def weight_sweep(df: pd.DataFrame, weights: list[float], budget_eur: float, cost_per_customer: float, max_customers: int, save_rate: float):
    rows = []
    for w_loss in weights:
        w_clv = 1 - w_loss
        blended = build_blended_score(df, w_loss, w_clv)
        _, s = optimize_targeting(blended, budget_eur, cost_per_customer, max_customers, save_rate, "blended_score")
        rows.append({"w_loss": w_loss, "w_clv": w_clv, **s})
    return pd.DataFrame(rows)
