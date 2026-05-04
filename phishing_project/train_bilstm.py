"""
BiLSTM + GloVe + Self-Attention Phishing Tespit Modeli Eğitimi
================================================================
Mimari: Dual Branch (GloVe frozen + BiLSTM trainable + Self-Attention)
Veri: 18,650 e-posta (9325 phishing + 9325 safe)
maxlen: 200 token | GloVe: 100d
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense, Dropout,
    GlobalAveragePooling1D, Concatenate, Multiply, Permute,
    RepeatVector, Flatten, Activation, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

# ─── Konfigürasyon ───
MAX_SEQUENCE_LENGTH = 200
GLOVE_DIM = 100
VOCAB_SIZE = 50000       # Tokenizer max kelime
LSTM_UNITS = 128
DENSE_UNITS = 128
DROPOUT_RATE = 0.4
BATCH_SIZE = 64
EPOCHS = 15
RANDOM_STATE = 42

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "final_dataset.csv")
GLOVE_PATH = os.path.join(os.path.dirname(__file__), "data", "glove.6B.100d.txt")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "LSTM", "bilstm_model.keras")
TOKENIZER_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "LSTM", "tokenizer.pkl")
HISTORY_SAVE_PATH = os.path.join(os.path.dirname(__file__), "data", "bilstm_history.npy")

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ─── 1. Veri Yükleme ───
def load_data():
    print("📊 Veri yükleniyor...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["body", "label"])
    df["body"] = df["body"].astype(str)
    print(f"   Toplam: {len(df)} | Phishing: {df['label'].sum()} | Safe: {(df['label']==0).sum()}")
    return df


# ─── 2. Tokenization ───
def tokenize_data(texts, fit=True):
    """Metinleri token indekslerine çevirir ve padding uygular."""
    global tokenizer

    if fit:
        print(f"🔤 Tokenizer oluşturuluyor (max {VOCAB_SIZE} kelime)...")
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        word_index = tokenizer.word_index
        print(f"   Toplam benzersiz kelime: {len(word_index)}")
        print(f"   Kullanılan kelime sayısı: {min(len(word_index), VOCAB_SIZE)}")

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    print(f"   Padded shape: {padded.shape}")
    return padded


# ─── 3. GloVe Embedding Matrix ───
def build_glove_matrix():
    """GloVe vektörlerinden embedding matrisi oluşturur."""
    print(f"🌐 GloVe yükleniyor: {GLOVE_PATH}")

    if not os.path.exists(GLOVE_PATH):
        print("❌ GloVe dosyası bulunamadı! Önce download_glove.py çalıştırın.")
        sys.exit(1)

    # GloVe vektörlerini oku
    glove_index = {}
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_index[word] = vector

    print(f"   GloVe kelime sayısı: {len(glove_index)}")

    # Embedding matrisi oluştur
    word_index = tokenizer.word_index
    num_words = min(len(word_index) + 1, VOCAB_SIZE + 1)
    embedding_matrix = np.zeros((num_words, GLOVE_DIM))

    found = 0
    for word, i in word_index.items():
        if i >= num_words:
            continue
        vec = glove_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
            found += 1

    coverage = found / min(len(word_index), VOCAB_SIZE) * 100
    print(f"   GloVe kapsama: {found}/{min(len(word_index), VOCAB_SIZE)} kelime ({coverage:.1f}%)")
    return embedding_matrix, num_words


# ─── 4. BiLSTM + GloVe + Attention Model ───
def build_bilstm_model(embedding_matrix, num_words):
    """Dual Branch: GloVe (frozen) + BiLSTM (trainable) + Self-Attention"""
    print("🏗️ BiLSTM + GloVe + Attention modeli oluşturuluyor...")

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), name="input_sequence")

    # ═══ GloVe Branch (Frozen) ═══
    glove_emb = Embedding(
        input_dim=num_words,
        output_dim=GLOVE_DIM,
        weights=[embedding_matrix],
        trainable=False,
        name="glove_embedding"
    )(inp)
    glove_out = GlobalAveragePooling1D(name="glove_pool")(glove_emb)
    # Output: (batch, 100)

    # ═══ BiLSTM Branch (Trainable) ═══
    lstm_emb = Embedding(
        input_dim=num_words,
        output_dim=GLOVE_DIM,
        trainable=True,
        name="lstm_embedding"
    )(inp)

    # BiLSTM Layer
    lstm_out = Bidirectional(
        LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name="bilstm"
    )(lstm_emb)
    # Output: (batch, 200, 256)

    # Self-Attention (Bahdanau-style)
    attention_scores = Dense(1, activation='tanh', name="attention_dense")(lstm_out)
    attention_scores = Flatten(name="attention_flatten")(attention_scores)
    attention_weights = Activation('softmax', name="attention_softmax")(attention_scores)
    attention_weights = RepeatVector(LSTM_UNITS * 2, name="attention_repeat")(attention_weights)
    attention_weights = Permute([2, 1], name="attention_permute")(attention_weights)

    lstm_attended = Multiply(name="attention_multiply")([lstm_out, attention_weights])
    lstm_context = Lambda(lambda x: K.sum(x, axis=1), name="attention_sum")(lstm_attended)
    # Output: (batch, 256)

    # ═══ Fusion ═══
    merged = Concatenate(name="fusion")([glove_out, lstm_context])
    # Output: (batch, 356)

    x = Dense(DENSE_UNITS, activation='relu', name="dense_1")(merged)
    x = Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = Dense(64, activation='relu', name="dense_2")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    output = Dense(1, activation='sigmoid', name="output")(x)

    model = Model(inputs=inp, outputs=output, name="BiLSTM_GloVe_Attention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    model.summary()
    return model


# ─── 5. Eğitim ───
def train_model(model, X_train, y_train, X_val, y_val):
    """Model eğitimi — EarlyStopping + ReduceLR + Checkpoint."""
    print(f"\n🚀 Eğitim başlıyor ({EPOCHS} epoch, batch={BATCH_SIZE})...")

    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return history


# ─── 6. Değerlendirme ───
def evaluate_model(model, X_test, y_test):
    """Test seti üzerinde kapsamlı değerlendirme."""
    print("\n📈 Test değerlendirmesi...")

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("  BiLSTM + GloVe + Attention — TEST SONUÇLARI")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print("=" * 50)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


# ─── 7. Ana Akış ───
def main():
    print("=" * 60)
    print("  BiLSTM + GloVe + Attention — Phishing Model Eğitimi")
    print("=" * 60)

    # 1. Veri yükle
    df = load_data()

    # 2. Train/Val/Test bölme (70/15/15)
    texts = df["body"].tolist()
    labels = df["label"].values

    X_temp, X_test_text, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=RANDOM_STATE
    )
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=RANDOM_STATE
    )
    # 0.176 * 0.85 ≈ 0.15

    print(f"\n📂 Veri bölümü:")
    print(f"   Train: {len(X_train_text)} ({len(X_train_text)/len(texts)*100:.1f}%)")
    print(f"   Val  : {len(X_val_text)} ({len(X_val_text)/len(texts)*100:.1f}%)")
    print(f"   Test : {len(X_test_text)} ({len(X_test_text)/len(texts)*100:.1f}%)")

    # 3. Tokenization + Padding
    all_texts = X_train_text + X_val_text + X_test_text
    tokenize_data(all_texts, fit=True)

    X_train = tokenize_data(X_train_text, fit=False)
    X_val = tokenize_data(X_val_text, fit=False)
    X_test = tokenize_data(X_test_text, fit=False)

    # 4. Tokenizer kaydet
    os.makedirs(os.path.dirname(TOKENIZER_SAVE_PATH), exist_ok=True)
    with open(TOKENIZER_SAVE_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"💾 Tokenizer kaydedildi: {TOKENIZER_SAVE_PATH}")

    # 5. GloVe embedding matrix
    embedding_matrix, num_words = build_glove_matrix()

    # 6. Model oluştur
    model = build_bilstm_model(embedding_matrix, num_words)

    # 7. Eğit
    history = train_model(model, X_train, y_train, X_val, y_val)

    # 8. Değerlendir
    metrics = evaluate_model(model, X_test, y_test)

    # 9. History kaydet
    np.save(HISTORY_SAVE_PATH, history.history)
    print(f"💾 Eğitim geçmişi kaydedildi: {HISTORY_SAVE_PATH}")

    # 10. Özet
    print("\n" + "=" * 60)
    print("  ✅ EĞİTİM TAMAMLANDI")
    print("=" * 60)
    print(f"  Model      : {MODEL_SAVE_PATH}")
    print(f"  Tokenizer  : {TOKENIZER_SAVE_PATH}")
    print(f"  maxlen     : {MAX_SEQUENCE_LENGTH}")
    print(f"  GloVe      : {GLOVE_DIM}d")
    print(f"  F1-Score   : {metrics['f1']:.4f}")
    print(f"  ROC-AUC    : {metrics['auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
