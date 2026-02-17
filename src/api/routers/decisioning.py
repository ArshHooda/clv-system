from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_db
from src.api.schemas import DecisioningRequest
from src.clv.decisioning import build_blended_score, optimize_targeting

router = APIRouter(prefix="/decisioning", tags=["decisioning"])


@router.post("/simulate")
def simulate(req: DecisioningRequest, con=Depends(get_db)):
    pred = con.execute("SELECT * FROM predictions_customer_latest").fetch_df()
    top_loss, sum_loss = optimize_targeting(pred, req.budget_eur, req.cost_per_customer, req.max_customers, req.save_rate, "expected_loss")
    blended = build_blended_score(pred, req.w_loss, req.w_clv)
    top_blend, sum_blend = optimize_targeting(blended, req.budget_eur, req.cost_per_customer, req.max_customers, req.save_rate, "blended_score")
    overlap = len(set(top_loss.CustomerID).intersection(set(top_blend.CustomerID))) / max(1, len(top_loss))
    return {
        "loss_only": {"summary": sum_loss, "top_n": top_loss.head(req.top_n).to_dict(orient="records")},
        "blended": {"summary": sum_blend, "top_n": top_blend.head(req.top_n).to_dict(orient="records")},
        "overlap_pct": overlap,
    }
