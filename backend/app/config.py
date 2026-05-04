"""
Sistem konfigürasyonu — Model yolları, eşik değerleri, sabitler.
"""
import os
from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Ağ ödevi/

# Model dosya yolları
XGBOOST_MODEL_PATH = BASE_DIR / "phishing_project" / "data" / "xgboost_model.pkl"
SCALER_PATH = BASE_DIR / "phishing_project" / "data" / "scaler.pkl"
LSTM_MODEL_PATH = BASE_DIR / "LSTM" / "bilstm_model.keras"
TOKENIZER_PATH = BASE_DIR / "LSTM" / "tokenizer.pkl"

# Adaptif eşik değerleri (Youden-J ile kalibre edilecek)
THRESHOLD_LOW = float(os.getenv("THRESHOLD_LOW", "0.35"))
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", "0.65"))

# LSTM parametreleri
LSTM_MAX_SEQUENCE_LENGTH = 200  # BiLSTM eğitimindeki maxlen

# Stilometrik feature sütunları (mevcut 13)
FEATURE_COLS = [
    "num_chars", "num_words", "num_sentences",
    "num_upper_chars", "num_special_chars",
    "avg_word_size", "words_to_chars",
    "special_chars_to_chars", "unique_words_to_word",
    "exclamationmark_to_chars", "questionmark_to_chars",
    "avg_words_in_sentence", "max_words_in_sentence"
]

# Eylem eşleme
ACTION_MAP = {
    (0.80, 1.01): ("PHISHING", "QUARANTINE"),
    (0.65, 0.80): ("PHISHING", "SPAM_FOLDER"),
    (0.35, 0.65): ("UNCERTAIN", "MANUAL_REVIEW"),
    (0.00, 0.35): ("SAFE", "INBOX"),
}

# API ayarları
API_VERSION = "3.0.0"
API_TITLE = "Phishing Detection AI Agent System"
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000",
                "http://localhost:5500", "http://127.0.0.1:5500"]
