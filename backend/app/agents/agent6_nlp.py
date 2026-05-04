"""
Agent 6 — NLP Preprocessing Agent
UNCERTAIN e-postalar için derin NLP ön işleme: tokenize + padding.
"""
import re
import pickle
import numpy as np
from app.config import TOKENIZER_PATH, LSTM_MAX_SEQUENCE_LENGTH

_tokenizer = None


def _load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with open(TOKENIZER_PATH, 'rb') as f:
            _tokenizer = pickle.load(f)
    return _tokenizer


def is_loaded() -> bool:
    try:
        _load_tokenizer()
        return True
    except Exception:
        return False


def _clean_for_nlp(text: str) -> str:
    """NLP pipeline için metin temizleme (stilometrikten farklı!)."""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def nlp_preprocess(text: str) -> dict:
    """
    NLP ön işleme:
    1. Temizleme (lowercase, URL/HTML kaldırma)
    2. Tokenization (kelime → index)
    3. Sequence Padding (maxlen=200)
    """
    # Keras 3 uyumlu pad_sequences — tf.keras veya keras.preprocessing
    try:
        from keras.src.utils.sequence_utils import pad_sequences
    except ImportError:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError:
            # Manuel numpy fallback
            def pad_sequences(seqs, maxlen):
                out = np.zeros((len(seqs), maxlen), dtype='int32')
                for i, s in enumerate(seqs):
                    s = s[:maxlen]
                    out[i, :len(s)] = s
                return out

    tokenizer = _load_tokenizer()
    cleaned = _clean_for_nlp(text)

    # Tokenize
    sequences = tokenizer.texts_to_sequences([cleaned])

    # Padding
    padded = pad_sequences(sequences, maxlen=LSTM_MAX_SEQUENCE_LENGTH)

    tokens = cleaned.split()
    return {
        "cleaned_text": cleaned,
        "tokens": tokens[:20],  # İlk 20 token (debug için)
        "sequence_length": len(sequences[0]) if sequences[0] else 0,
        "padded_sequence": padded,
    }
