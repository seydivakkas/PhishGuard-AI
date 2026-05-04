"""
Health Router — Sistem sağlık kontrolü ve eşik yönetimi.
"""
from fastapi import APIRouter
from app.schemas.response import HealthResponse
from app.config import API_VERSION
from app.agents.agent5_threshold import get_thresholds, update_thresholds
from pydantic import BaseModel

router = APIRouter()


class ThresholdUpdate(BaseModel):
    t_low: float
    t_high: float


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Sistem sağlık durumu: model yüklü mü, eşik değerleri, versiyon."""
    from app.agents.agent4_xgboost import is_loaded as xgb_loaded
    from app.agents.agent6_nlp import is_loaded as tok_loaded
    from app.agents.agent7_dual_branch import is_loaded as lstm_loaded

    thresholds = get_thresholds()
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        xgboost_loaded=xgb_loaded(),
        lstm_loaded=lstm_loaded(),
        scaler_loaded=xgb_loaded(),
        tokenizer_loaded=tok_loaded(),
        threshold_low=thresholds["t_low"],
        threshold_high=thresholds["t_high"],
    )


@router.patch("/threshold")
async def update_threshold(body: ThresholdUpdate):
    """Adaptif eşikleri runtime'da güncelle."""
    update_thresholds(body.t_low, body.t_high)
    return {"status": "updated", **get_thresholds()}
