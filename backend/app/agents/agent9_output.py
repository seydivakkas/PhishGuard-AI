"""
Agent 9 — Output & Logging Agent
Nihai kararı formatlar, loglar ve downstream sistemlere iletir.
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger("phishing.output")


def determine_action(probability: float, verdict: str) -> str:
    """Skor → Aksiyon eşlemesi."""
    if verdict == "PHISHING":
        return "QUARANTINE" if probability >= 0.80 else "SPAM_FOLDER"
    elif verdict == "SAFE":
        return "INBOX"
    else:
        return "MANUAL_REVIEW"


def build_explanation(verdict: str, probability: float, stylo_features: dict) -> str:
    """İnsan-okunabilir açıklama üretir."""
    parts = []

    if verdict == "PHISHING":
        parts.append(f"Bu e-posta %{probability*100:.1f} olasılıkla phishing olarak tespit edildi.")
        # Stilometrik ipuçları
        if stylo_features.get("num_upper_chars", 0) > 50:
            parts.append("Yüksek büyük harf kullanımı tespit edildi.")
        if stylo_features.get("exclamationmark_to_chars", 0) > 0.01:
            parts.append("Aşırı ünlem işareti kullanımı mevcut.")
        if stylo_features.get("num_special_chars", 0) > 30:
            parts.append("Anormal özel karakter yoğunluğu.")
    elif verdict == "SAFE":
        parts.append(f"Bu e-posta %{(1-probability)*100:.1f} güvenle safe olarak sınıflandırıldı.")
        parts.append("Normal yazım stili ve içerik örüntüsü tespit edildi.")
    else:
        parts.append("Otomatik karar verilemiyor — manuel inceleme önerilir.")

    return " ".join(parts) if parts else "Analiz tamamlandı."


async def format_output(email_id: str, verdict: str, probability: float,
                        xgb_score: float, lstm_score: float = None,
                        detection_path: str = "FAST_PATH",
                        processing_time_ms: float = 0,
                        stylo_features: dict = None) -> dict:
    """Nihai çıktıyı formatlar ve loglar."""

    action = determine_action(probability, verdict)
    explanation = build_explanation(verdict, probability, stylo_features or {})

    if probability > 0.85 or probability < 0.15:
        confidence = "HIGH"
    elif probability > 0.7 or probability < 0.3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    result = {
        "email_id": email_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "final_probability": round(probability, 4),
        "xgboost_score": round(xgb_score, 4),
        "lstm_score": round(lstm_score, 4) if lstm_score is not None else None,
        "detection_path": detection_path,
        "processing_time_ms": round(processing_time_ms, 2),
        "confidence_level": confidence,
        "action": action,
        "explanation": explanation,
        "stylometric_features": stylo_features,
    }

    # Log
    logger.info(
        f"[{verdict}] email={email_id} prob={probability:.4f} "
        f"path={detection_path} latency={processing_time_ms:.1f}ms action={action}"
    )

    return result
