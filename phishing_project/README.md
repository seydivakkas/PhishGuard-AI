# Phishing Email Detection System

Bu proje, e-posta içeriklerinden phishing (oltalama) ve normal (ham) e-postaları tespit eden hibrit bir makine öğrenmesi sistemidir.

Sistem iki ana bileşenden oluşur:
- Feature-based model (XGBoost)
- Stilometrik + istatistiksel özellik çıkarımı
- LSTM ile hibrit kullanımı

---
## Kurulum

1. Seçenek 1 — Birebir aynı ortam 
   pip install -r requirements.txt

2. conda env create -f environment.yml
   conda activate phishing_env

## NLTK setup
python -m nltk.downloader stopwords punkt

## Çalıştırma

Pipeline sırası:

- python preprocess.py
- python feature_extraction.py
- python split_data.py
- python train_xgb.py



##  Dataset

Proje aşağıdaki veri kaynaklarını birleştirir:

- CEAS_08
- Enron
- SpamAssassin
- Ling
- Nazario
- Nigerian Fraud

Toplam veri: **18,650 email**

---

##  Özellik Çıkarma

Her e-posta için aşağıdaki stilometrik özellikler çıkarılır:

1. Temel boyut bilgisi :
- num_chars 
- num_words 
- num_sentences
- num_upper_chars
- num_special_chars

2. Stil / Yazım Özellikleri  :
- avg_word_size

3. Oranlar :
- words_to_chars
- special_chars_to_chars
- unique_words_to_word  
- exclamationmark_to_chars
- questionmark_to_chars

4. Cümle istatistikleri:
- avg_words_in_sentence 
- max_words_in_sentence  
---

##  Model

Kullanılan model:
- XGBoost Classifier

Parametreler:
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- eval_metric= logloss
- early_stopping_rounds=20
- random_state: 42 



##  Proje Yapısı

phishing_project/
├── preprocess.py
├── feature_extraction.py
├── split_data.py
├── train_xgb.py
├── README.md
├── requirements.txt
├── environment.yml
└── data/
      ├── final_dataset.csv
      ├── featured_dataset.csv
      ├── X_train.npy
      ├── X_val.npy
      ├── X_test.npy
      ├── y_train.npy
      ├── y_val.npy
      ├── y_test.npy
      ├── scaler.pkl
      ├── xgboost_model.pkl
      ├── train_idx.npy
      ├── val_idx.npy
      └── test_idx.npy

##  Çalıştırma Adımları

### 1. Veri hazırlama

python preprocess.py


### 2. Feature extraction

python feature_extraction.py


### 3. Train / validation / test split

python split_data.py


### 4. Model eğitimi

python train_xgb.py
---

##   XGBOOST Model Performansı

- Accuracy: ~0.88
- Precision: ~0.88
- Recall: ~0.88
- F1-score: ~0.88

---

##  Çıktılar

Model çalıştırıldıktan sonra:
- feature_importance.png
- xgboost_model.pkl dosyaları oluşturulur.

--- # lstm ve diğer kısımları devam ettirirsiniz.
