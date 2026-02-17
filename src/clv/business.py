import pandas as pd


def retention_simulation(df: pd.DataFrame, target_pct: float, save_rate: float, cost_per_customer: float):
    """
    df must include: CustomerID, churn_prob, revenue_pred_window, risk_score
    target_pct: 0.1 / 0.2 / 0.3 ...
    save_rate: expected fraction of targeted churners you retain (0-1)
    cost_per_customer: campaign cost per targeted customer
    """
    df = df.sort_values("risk_score", ascending=False).copy()
    top_n = int(len(df) * target_pct)
    targeted = df.head(top_n).copy()

    # Expected prevented loss = save_rate * sum(risk_score)
    expected_prevented_revenue = save_rate * targeted["risk_score"].sum()

    total_cost = cost_per_customer * len(targeted)
    net_uplift = expected_prevented_revenue - total_cost
    roi = (net_uplift / total_cost) if total_cost > 0 else None

    return {
        "targeted_customers": len(targeted),
        "expected_prevented_revenue": float(expected_prevented_revenue),
        "total_cost": float(total_cost),
        "net_uplift": float(net_uplift),
        "roi": None if roi is None else float(roi),
        "target_list": targeted[["CustomerID", "churn_prob", "revenue_pred_window", "risk_score"]],
    }

def optimize_targeting(
    df: pd.DataFrame,
    budget_eur: float,
    cost_per_customer: float,
    max_customers: int,
    save_rate: float,
    score_col: str = "expected_loss",
):
    """
    df must include: CustomerID and score_col (default expected_loss)
    Optional: churn_prob, spend_prob, expected_revenue, expected_clv (for reporting)

    Select top customers by score_col under BOTH:
    - budget constraint
    - max_customers constraint
    """

    if cost_per_customer <= 0:
        raise ValueError("cost_per_customer must be > 0")
    if budget_eur < 0:
        raise ValueError("budget_eur must be >= 0")
    if max_customers <= 0:
        raise ValueError("max_customers must be > 0")
    if not (0 <= save_rate <= 1):
        raise ValueError("save_rate must be between 0 and 1")

    df = df.copy().sort_values(score_col, ascending=False)

    # Capacity constraint first
    df = df.head(max_customers).copy()

    # Budget constraint: how many can we afford?
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

