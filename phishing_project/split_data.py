import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


df = pd.read_csv("data/featured_dataset.csv")
print(f"Yüklendi: {len(df)} satır")

FEATURE_COLS = [
    "num_chars", "num_words", "num_sentences",
    "num_upper_chars", "num_special_chars",
    "avg_word_size", "words_to_chars",
    "special_chars_to_chars", "unique_words_to_word",
    "exclamationmark_to_chars", "questionmark_to_chars",
    "avg_words_in_sentence", "max_words_in_sentence"
]

X = df[FEATURE_COLS]
y = df["label"]

# Önce %70 train
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

#  Kalan % 30'lık kısmı  %15 val, %15 test olarak böl
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print(f"\nTrain : {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Val   : {len(X_val)}   ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test  : {len(X_test)}  ({len(X_test)/len(df)*100:.1f}%)")

# Sınıf dağılımı kontrolü
print(f"\nTrain label dağılımı:\n{y_train.value_counts()}")
print(f"\nVal label dağılımı:\n{y_val.value_counts()}")
print(f"\nTest label dağılımı:\n{y_test.value_counts()}")



#Scale etme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #sadece burda fit var
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Index kaydı
np.save("data/train_idx.npy", X_train.index.values)
np.save("data/val_idx.npy",   X_val.index.values)
np.save("data/test_idx.npy",  X_test.index.values)


np.save("data/X_train.npy", X_train_scaled)
np.save("data/X_val.npy",   X_val_scaled)
np.save("data/X_test.npy",  X_test_scaled)
np.save("data/y_train.npy", y_train.values)
np.save("data/y_val.npy",   y_val.values)
np.save("data/y_test.npy",  y_test.values)

joblib.dump(scaler, "data/scaler.pkl")

print("\n--- KAYDEDILENLER ---")
print("data/X_train.npy, X_val.npy, X_test.npy")
print("data/y_train.npy, y_val.npy, y_test.npy")
print("data/train_idx.npy, val_idx.npy, test_idx.npy")
print("data/scaler.pkl")
print("\nTamamlandı ✅")

# Kaynak dağılımı kontrolü
df_train = df.loc[X_train.index]
df_val   = df.loc[X_val.index]
df_test  = df.loc[X_test.index]

print("\nTrain kaynak dağılımı:")
print(df_train["source"].value_counts())
print("\nVal kaynak dağılımı:")
print(df_val["source"].value_counts())
print("\nTest kaynak dağılımı:")
print(df_test["source"].value_counts())