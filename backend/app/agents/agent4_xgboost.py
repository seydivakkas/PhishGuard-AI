"""
Agent 4 — XGBoost Classifier Agent
Stilometrik feature vektörünü alarak XGBoost modeli ile phishing olasılık skoru üretir.
"""
import numpy as np
import joblib
from app.config import XGBOOST_MODEL_PATH, SCALER_PATH

# Model ve scaler lazy-load
_model = None
_scaler = None


def _load_models():
    global _model, _scaler
    if _model is None:
        _model = joblib.load(XGBOOST_MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def is_loaded() -> bool:
    try:
        _load_models()
        return True
    except Exception:
        return False


async def xgboost_predict(feature_vector: list) -> dict:
    """
    XGBoost ile phishing olasılık skoru üretir.
    Input: 13-dim stilometrik feature vektörü
    Output: P(phishing) ∈ [0.0, 1.0]
    """
    model, scaler = _load_models()

    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X)

    probability = float(model.predict_proba(X_scaled)[0][1])
    prediction = int(model.predict(X_scaled)[0])

    # Feature importance (top 3)
    importance = model.feature_importances_
    from app.config import FEATURE_COLS
    top_indices = np.argsort(importance)[::-1][:3]
    top_features = {
        FEATURE_COLS[i]: round(float(importance[i]), 4) for i in top_indices
    }

    return {
        "probability": round(probability, 4),
        "prediction": prediction,
        "top_features": top_features,
        "model_confidence": "HIGH" if abs(probability - 0.5) > 0.3 else "MEDIUM" if abs(probability - 0.5) > 0.15 else "LOW",
    }
