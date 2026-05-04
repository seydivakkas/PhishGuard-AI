"""Pydantic response modelleri."""
from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    email_id: str
    verdict: str                        # "PHISHING" | "SAFE" | "UNCERTAIN"
    final_probability: float
    xgboost_score: float
    lstm_score: Optional[float] = None
    detection_path: str                 # "FAST_PATH" | "FULL_PIPELINE"
    processing_time_ms: float
    confidence_level: str               # "HIGH" | "MEDIUM" | "LOW"
    action: str                         # "QUARANTINE" | "INBOX" | "MANUAL_REVIEW" | "SPAM_FOLDER"
    explanation: str
    stylometric_features: Optional[dict] = None


class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    mcc: float
    false_negative_rate: float
    avg_latency_ms: float
    throughput_rps: float
    total_samples: int
    phishing_count: int
    safe_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    xgboost_loaded: bool
    lstm_loaded: bool
    scaler_loaded: bool
    tokenizer_loaded: bool
    threshold_low: float
    threshold_high: float
