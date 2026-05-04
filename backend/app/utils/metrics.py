"""Performans metrikleri hesaplama."""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix
)
import numpy as np


def compute_metrics(y_true: list, y_pred: list, latencies: list = None) -> dict:
    """10 performans metriği hesaplar."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    lats = latencies or [0]
    avg_lat = float(np.mean(lats))

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0, 4),
        "pr_auc": round(average_precision_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0, 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0,
        "avg_latency_ms": round(avg_lat, 2),
        "throughput_rps": round(1000 / avg_lat, 2) if avg_lat > 0 else 0.0,
        "total_samples": len(y_true),
        "phishing_count": int(sum(y_true)),
        "safe_count": int(len(y_true) - sum(y_true)),
    }
