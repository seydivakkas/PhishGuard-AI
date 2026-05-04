"""
Agent 8 — Meta-Classifier / Stacking Agent
XGBoost (stylometric) ve LSTM (content) çıktılarını Stacking ile birleştirir.
Level-2: Son nihai karar.
"""


async def meta_classify(xgb_prob: float, lstm_prob: float) -> dict:
    """
    Stacking meta-classifier:
    Meta-features: [P1, P2, P1×P2, |P1−P2|, (P1+P2)/2]

    Basit ağırlıklı ortalama + disagreement analizi.
    (Tam Logistic Regression meta-model eğitildikten sonra değiştirilebilir.)
    """
    # Meta-feature mühendisliği
    product = xgb_prob * lstm_prob
    difference = abs(xgb_prob - lstm_prob)
    average = (xgb_prob + lstm_prob) / 2

    # Ağırlıklı birleştirme (LSTM'e biraz daha fazla ağırlık — derin analiz)
    # XGBoost: 0.4, LSTM: 0.6
    weighted = xgb_prob * 0.4 + lstm_prob * 0.6

    # Disagreement durumunda ortalamaya yaklaş
    if difference > 0.3:
        final_prob = average  # Yüksek uyumsuzluk → ortalama al
        agreement = "LOW"
    elif difference > 0.15:
        final_prob = weighted
        agreement = "MEDIUM"
    else:
        final_prob = weighted
        agreement = "HIGH"

    final_prediction = 1 if final_prob >= 0.5 else 0

    return {
        "prediction": final_prediction,
        "probability": round(final_prob, 4),
        "meta_features": {
            "xgb_prob": round(xgb_prob, 4),
            "lstm_prob": round(lstm_prob, 4),
            "product": round(product, 4),
            "difference": round(difference, 4),
            "average": round(average, 4),
        },
        "model_agreement": agreement,
    }
