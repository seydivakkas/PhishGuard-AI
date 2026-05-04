import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)


X_train = np.load("data/X_train.npy")
X_val   = np.load("data/X_val.npy")
X_test  = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_val   = np.load("data/y_val.npy")
y_test  = np.load("data/y_test.npy")

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


model = XGBClassifier(
 n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
    verbosity=1
)
#Eğitim
print("\nModel eğitiliyor...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

#Tahmin
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

#Sonuçlar
print("\n--- TEST SONUÇLARI ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Hibrit model için val seti olasılıkları
y_val_proba = model.predict_proba(X_val)[:, 1]

np.save("data/xgb_test_proba.npy", y_pred_proba)
np.save("data/xgb_val_proba.npy",  y_val_proba)


joblib.dump(model, "data/xgboost_model.pkl")
print("\nModel kaydedildi: data/xgboost_model.pkl")

#Feature Importance 
FEATURE_COLS = [
    "num_chars", "num_words", "num_sentences",
    "num_upper_chars", "num_special_chars",
    "avg_word_size", "words_to_chars",
    "special_chars_to_chars", "unique_words_to_word",
    "exclamationmark_to_chars", "questionmark_to_chars",
    "avg_words_in_sentence", "max_words_in_sentence"
]

importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(FEATURE_COLS)), importance[indices])
plt.xticks(range(len(FEATURE_COLS)),
           [FEATURE_COLS[i] for i in indices],
           rotation=45, ha="right")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("data/feature_importance.png")
print("Feature importance kaydedildi: data/feature_importance.png")