"""
Agent 7 — Dual Branch Embedding Agent (GloVe + BiLSTM + Attention)
Eğitilmiş BiLSTM modeli (bilstm_model.keras) ile tahmin yapar.
Mimari: GloVe frozen embedding + BiLSTM trainable + Self-Attention
"""
import numpy as np
from app.config import LSTM_MODEL_PATH

_model = None

# Eğitimdeki sabitler
MAX_SEQUENCE_LENGTH = 200
GLOVE_DIM = 100
LSTM_UNITS = 128
DENSE_UNITS = 128
DROPOUT_RATE = 0.4


def _rebuild_model():
    """Modeli mimari tanımıyla yeniden oluştur ve weights yükle."""
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, Embedding, Bidirectional, LSTM, Dense, Dropout,
        GlobalAveragePooling1D, Concatenate, Multiply, Permute,
        RepeatVector, Flatten, Activation, Lambda
    )
    from tensorflow.keras.models import Model
    import tensorflow.keras.backend as K
    import zipfile, io, json

    # .keras dosyasından num_words bilgisini çıkar
    with zipfile.ZipFile(str(LSTM_MODEL_PATH), 'r') as z:
        cfg = json.loads(z.read('config.json'))
    # glove_embedding layer config'inden input_dim al
    for layer_cfg in cfg['config']['layers']:
        if layer_cfg.get('name') == 'glove_embedding':
            num_words = layer_cfg['config']['input_dim']
            break
    else:
        num_words = 50001  # fallback

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), name="input_sequence")

    # GloVe Branch (Frozen)
    glove_emb = Embedding(num_words, GLOVE_DIM, trainable=False, name="glove_embedding")(inp)
    glove_out = GlobalAveragePooling1D(name="glove_pool")(glove_emb)

    # BiLSTM Branch (Trainable)
    lstm_emb = Embedding(num_words, GLOVE_DIM, trainable=True, name="lstm_embedding")(inp)
    lstm_out = Bidirectional(
        LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name="bilstm"
    )(lstm_emb)

    # Self-Attention
    attention_scores = Dense(1, activation='tanh', name="attention_dense")(lstm_out)
    attention_scores = Flatten(name="attention_flatten")(attention_scores)
    attention_weights = Activation('softmax', name="attention_softmax")(attention_scores)
    attention_weights = RepeatVector(LSTM_UNITS * 2, name="attention_repeat")(attention_weights)
    attention_weights = Permute([2, 1], name="attention_permute")(attention_weights)

    lstm_attended = Multiply(name="attention_multiply")([lstm_out, attention_weights])
    lstm_context = Lambda(
        lambda x: K.sum(x, axis=1),
        output_shape=lambda s: (s[0], s[2]),
        name="attention_sum"
    )(lstm_attended)

    # Fusion
    merged = Concatenate(name="fusion")([glove_out, lstm_context])
    x = Dense(DENSE_UNITS, activation='relu', name="dense_1")(merged)
    x = Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = Dense(64, activation='relu', name="dense_2")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    output = Dense(1, activation='sigmoid', name="output")(x)

    model = Model(inputs=inp, outputs=output, name="BiLSTM_GloVe_Attention")

    # Weights yükle (.keras zip dosyasından)
    model.load_weights(str(LSTM_MODEL_PATH))

    return model


def _load_model():
    global _model
    if _model is None:
        _model = _rebuild_model()
    return _model


def is_loaded() -> bool:
    try:
        _load_model()
        return True
    except Exception:
        return False


async def dual_branch_predict(padded_sequence: np.ndarray) -> dict:
    """
    BiLSTM + GloVe + Attention modeli ile phishing olasılık skoru üretir.
    Input: padded_sequence [1, max_len=200]
    Output: P(phishing) ∈ [0.0, 1.0]
    """
    model = _load_model()

    prediction = model.predict(padded_sequence, verbose=0)
    probability = float(prediction[0][0])

    return {
        "probability": round(probability, 4),
        "prediction": 1 if probability >= 0.5 else 0,
        "model_type": "BiLSTM+GloVe+Attention",
    }
