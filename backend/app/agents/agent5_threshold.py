"""
Agent 5 — Adaptive Threshold Decision Agent
XGBoost olasılık skorunu 3 bölgeye ayırır: CERTAIN/PHISHING, UNCERTAIN, CERTAIN/SAFE.
Youden-J ile adaptif eşik kalibrasyonu destekler.
"""
from app.config import THRESHOLD_LOW, THRESHOLD_HIGH

# Mutable eşikler (runtime güncelleme için)
_current_thresholds = {
    "t_low": THRESHOLD_LOW,
    "t_high": THRESHOLD_HIGH,
}


def adaptive_threshold_decision(probability: float) -> dict:
    """
    Eşik bölgeleri:
    0.0 ────── [T_low=0.35] ────── [T_high=0.65] ────── 1.0
       CERTAIN/SAFE         UNCERTAIN         CERTAIN/PHISHING
    """
    t_low = _current_thresholds["t_low"]
    t_high = _current_thresholds["t_high"]

    if probability >= t_high:
        decision = "CERTAIN/PHISHING"
        next_agent = None  # Fast path — direkt çıkış
    elif probability <= t_low:
        decision = "CERTAIN/SAFE"
        next_agent = None  # Fast path — direkt çıkış
    else:
        decision = "UNCERTAIN"
        next_agent = "agent6_nlp"  # Derin analiz pipeline'a yönlendir

    return {
        "decision": decision,
        "probability": probability,
        "threshold_low": t_low,
        "threshold_high": t_high,
        "next_agent": next_agent,
    }


def update_thresholds(t_low: float, t_high: float):
    """Eşikleri runtime'da güncelle (Youden-J kalibrasyonu sonrası)."""
    _current_thresholds["t_low"] = t_low
    _current_thresholds["t_high"] = t_high


def get_thresholds() -> dict:
    return dict(_current_thresholds)
