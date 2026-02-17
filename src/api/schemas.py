from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    latest_cutoff: str | None
    rows_latest: int


class PredictionsSummary(BaseModel):
    rows: int
    avg_churn_prob: float
    avg_expected_loss: float
    avg_expected_clv: float


class DecisioningRequest(BaseModel):
    budget_eur: float = 500
    cost_per_customer: float = 1
    max_customers: int = 2000
    save_rate: float = 0.15
    w_loss: float = 0.7
    w_clv: float = 0.3
    top_n: int = 10
