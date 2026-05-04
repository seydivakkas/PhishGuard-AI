"""
Agent 3 — Stylometric Feature Extraction Agent
Metnin yazım stilini analiz eder. HAM METİNDEN çalışır (temizlenmemiş).
⚠️ Stopword removal, lemmatization YAPILMAZ — stil bilgisi korunur.
"""
import re


def extract_stylometric_features(raw_text: str) -> dict:
    """
    13 boyutlu stilometrik feature vektörü çıkarır.
    Girdi: Temizlenmemiş ham metin (stil bilgisi korunmalı).
    """
    text = str(raw_text)

    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]

    num_chars = len(text)
    num_words = len(words)
    num_sentences = max(len(sentences), 1)
    num_upper_chars = sum(1 for c in text if c.isupper())
    num_special = sum(1 for c in text if not c.isalnum() and not c.isspace())

    avg_word_size = (sum(len(w) for w in words) / num_words) if num_words > 0 else 0
    unique_words = len(set(w.lower() for w in words))

    words_to_chars = num_words / num_chars if num_chars > 0 else 0
    special_to_chars = num_special / num_chars if num_chars > 0 else 0
    unique_to_words = unique_words / num_words if num_words > 0 else 0
    exclamation_to_chars = text.count("!") / num_chars if num_chars > 0 else 0
    question_to_chars = text.count("?") / num_chars if num_chars > 0 else 0

    words_per_sent = [len(s.split()) for s in sentences]
    avg_words_in_sent = sum(words_per_sent) / len(words_per_sent) if words_per_sent else 0
    max_words_in_sent = max(words_per_sent) if words_per_sent else 0

    return {
        "num_chars": num_chars,
        "num_words": num_words,
        "num_sentences": num_sentences,
        "num_upper_chars": num_upper_chars,
        "num_special_chars": num_special,
        "avg_word_size": round(avg_word_size, 4),
        "words_to_chars": round(words_to_chars, 6),
        "special_chars_to_chars": round(special_to_chars, 6),
        "unique_words_to_word": round(unique_to_words, 4),
        "exclamationmark_to_chars": round(exclamation_to_chars, 6),
        "questionmark_to_chars": round(question_to_chars, 6),
        "avg_words_in_sentence": round(avg_words_in_sent, 4),
        "max_words_in_sentence": max_words_in_sent,
    }


def features_to_vector(features: dict) -> list:
    """Feature dict'i sıralı vektöre dönüştürür (XGBoost girişi için)."""
    from app.config import FEATURE_COLS
    return [features[col] for col in FEATURE_COLS]
