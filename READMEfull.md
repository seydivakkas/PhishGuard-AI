<div align="center">

# 🛡️ PhishGuard AI

### BiLSTM + XGBoost Tabanlı Hibrit Phishing Tespit Sistemi

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-189FDD?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=for-the-badge)](LICENSE)

**Accuracy: 96.93% · F1: 96.95% · ROC-AUC: 0.9944**

*9 Adımlı Adaptif Pipeline · Dual-Branch Stacking · Self-Attention*

---

**Geliştirici:** [seydieryilmazz](https://github.com/seydivakkas)

</div>

---

## 📑 İçindekiler

1. [Proje Özeti](#1-proje-özeti)
2. [Sistem Mimarisi](#2-sistem-mimarisi)
   - 2.1 [Genel Akış Diyagramı](#21-genel-akış-diyagramı)
   - 2.2 [Dallanmalı Pipeline (Hızlı Yol / Derin Analiz)](#22-dallanmalı-pipeline-hızlı-yol--derin-analiz)
   - 2.3 [Teknoloji Yığını](#23-teknoloji-yığını)
3. [Veri Setleri](#3-veri-setleri)
   - 3.1 [Kullanılan Veri Kaynakları](#31-kullanılan-veri-kaynakları)
   - 3.2 [Veri Ön İşleme (Preprocess)](#32-veri-ön-i̇şleme-preprocess)
   - 3.3 [Veri Bölme Stratejisi (Train / Val / Test)](#33-veri-bölme-stratejisi-train--val--test)
4. [Özellik Çıkarma (Feature Extraction)](#4-özellik-çıkarma-feature-extraction)
   - 4.1 [Stilometrik Özellikler (13 Feature)](#41-stilometrik-özellikler-13-feature)
   - 4.2 [Özellik Ölçekleme (StandardScaler)](#42-özellik-ölçekleme-standardscaler)
5. [Model Mimarisi](#5-model-mimarisi)
   - 5.1 [XGBoost Classifier (Branch 1)](#51-xgboost-classifier-branch-1)
   - 5.2 [BiLSTM + GloVe + Attention (Branch 2)](#52-bilstm--glove--attention-branch-2)
     - 5.2.1 [GloVe Embedding (Frozen)](#521-glove-embedding-frozen)
     - 5.2.2 [BiLSTM Katmanı (Trainable)](#522-bilstm-katmanı-trainable)
     - 5.2.3 [Self-Attention Mekanizması](#523-self-attention-mekanizması)
     - 5.2.4 [Fusion (Birleştirme)](#524-fusion-birleştirme)
   - 5.3 [Meta-Classifier (Stacking)](#53-meta-classifier-stacking)
   - 5.4 [Adaptif Eşik Mekanizması](#54-adaptif-eşik-mekanizması)
6. [NLP Pipeline](#6-nlp-pipeline)
   - 6.1 [Metin Temizleme Adımları](#61-metin-temizleme-adımları)
   - 6.2 [Tokenization](#62-tokenization)
   - 6.3 [Padding (maxlen=200)](#63-padding-maxlen200)
7. [9 Adımlı İşlem Pipeline'ı](#7-9-adımlı-i̇şlem-pipelineı)
   - 7.1 [Adım 1 — Ingestion](#71-adım-1--ingestion)
   - 7.2 [Adım 2 — Paralel Ön İşleme](#72-adım-2--paralel-ön-i̇şleme)
   - 7.3 [Adım 3 — Stilometrik Extraction](#73-adım-3--stilometrik-extraction)
   - 7.4 [Adım 4 — XGBoost Tahmin](#74-adım-4--xgboost-tahmin)
   - 7.5 [Adım 5 — Threshold (Eşik Karar)](#75-adım-5--threshold-eşik-karar)
   - 7.6 [Adım 6 — NLP Ön İşleme](#76-adım-6--nlp-ön-i̇şleme)
   - 7.7 [Adım 7 — BiLSTM + GloVe + Attention](#77-adım-7--bilstm--glove--attention)
   - 7.8 [Adım 8 — Meta-Classifier](#78-adım-8--meta-classifier)
   - 7.9 [Adım 9 — Sonuç Üretimi](#79-adım-9--sonuç-üretimi)
8. [Backend (FastAPI)](#8-backend-fastapi)
   - 8.1 [API Endpoint'leri](#81-api-endpointleri)
   - 8.2 [Proje Yapısı](#82-proje-yapısı)
   - 8.3 [Konfigürasyon (config.py)](#83-konfigürasyon-configpy)
   - 8.4 [Model Yükleme ve Servis](#84-model-yükleme-ve-servis)
9. [Frontend (Dashboard)](#9-frontend-dashboard)
   - 9.1 [Analiz Sayfası](#91-analiz-sayfası)
   - 9.2 [Değerlendirme Sayfası](#92-değerlendirme-sayfası)
     - 9.2.1 [Performans Metrikleri](#921-performans-metrikleri)
     - 9.2.2 [Veri Seti Analiz Havuzu](#922-veri-seti-analiz-havuzu)
   - 9.3 [Monitor Sayfası](#93-monitor-sayfası)
     - 9.3.1 [Adaptif Eşik Kontrolü](#931-adaptif-eşik-kontrolü)
     - 9.3.2 [Sistem Mimarisi (Dallanmalı Diyagram)](#932-sistem-mimarisi-dallanmalı-diyagram)
     - 9.3.3 [Sistem Sağlığı](#933-sistem-sağlığı)
   - 9.4 [Nasıl Çalışır Sayfası](#94-nasıl-çalışır-sayfası)
10. [Model Eğitimi](#10-model-eğitimi)
    - 10.1 [XGBoost Eğitimi](#101-xgboost-eğitimi)
    - 10.2 [BiLSTM Eğitimi](#102-bilstm-eğitimi)
    - 10.3 [Hiperparametre Tablosu](#103-hiperparametre-tablosu)
    - 10.4 [Eğitim Sonuçları](#104-eğitim-sonuçları)
11. [Performans Metrikleri](#11-performans-metrikleri)
    - 11.1 [XGBoost Sonuçları](#111-xgboost-sonuçları)
    - 11.2 [BiLSTM Sonuçları](#112-bilstm-sonuçları)
    - 11.3 [LSTM vs BiLSTM Karşılaştırması](#113-lstm-vs-bilstm-karşılaştırması)
12. [Kurulum ve Çalıştırma](#12-kurulum-ve-çalıştırma)
    - 12.1 [Gereksinimler](#121-gereksinimler)
    - 12.2 [Kurulum Adımları](#122-kurulum-adımları)
    - 12.3 [Tek Komutla Başlatma](#123-tek-komutla-başlatma)
13. [Proje Dosya Yapısı](#13-proje-dosya-yapısı)
14. [Kaynakça](#14-kaynakça)

---

## 1. Proje Özeti

**PhishGuard AI**, e-posta içeriklerinden phishing (oltalama) saldırılarını tespit eden hibrit bir derin öğrenme sistemidir. Sistem, geleneksel makine öğrenmesi (XGBoost) ile modern derin öğrenme (BiLSTM + GloVe + Self-Attention) yaklaşımlarını birleştiren **dual-branch stacking mimarisi** kullanır.

### Temel Özellikler

| Özellik | Değer |
|---|---|
| **Mimari** | Dual-Branch Stacking (XGBoost + BiLSTM) |
| **Veri Seti** | 18,650 e-posta (6 farklı kaynak) |
| **Accuracy** | %96.93 |
| **F1-Score** | %96.95 |
| **ROC-AUC** | 0.9944 |
| **Pipeline** | 9 Adımlı Adaptif Akış |
| **Backend** | FastAPI (Python 3.12) |
| **Frontend** | Vanilla HTML/CSS/JS Dashboard |

### Neden Hibrit Mimari?

Phishing e-postaları sürekli evrim geçirir. Tek bir modele güvenmek yetersiz kalır:

- **XGBoost** → Stilometrik özelliklere (yazım tarzı, karakter dağılımı) dayanır. Bilinen saldırı kalıplarını hızla yakalar.
- **BiLSTM** → E-posta metninin anlamsal bağlamını (context) öğrenir. Daha önce görülmemiş saldırıları bile tespit edebilir.
- **Meta-Classifier** → İki modelin çıktılarını birleştirerek her birinin zayıf yönlerini kompanse eder.

Sistem ayrıca **adaptif eşik mekanizması** ile kesin kararlar (SAFE/PHISHING) ile belirsiz vakaları (UNCERTAIN) ayırt eder. Belirsiz vakalar derin analize yönlendirilir, kesin olanlar hızlı yoldan geçer.

---

## 2. Sistem Mimarisi

### 2.1 Genel Akış Diyagramı

```
E-posta Girişi
      │
      ▼
┌─────────────┐
│  Adım 1     │  Ingestion (Konu + Gövde birleştirme)
│  Ingestion  │
└──────┬──────┘
       ▼
┌─────────────┐
│  Adım 2     │  Metin temizleme (lowercase, URL/email/HTML kaldırma)
│  Preprocess │
└──────┬──────┘
       ▼
┌─────────────┐
│  Adım 3     │  13 stilometrik özellik çıkarma
│  Stylometric│
└──────┬──────┘
       ▼
┌─────────────┐
│  Adım 4     │  XGBoost ile ilk tahmin → P₁ ∈ [0, 1]
│  XGBoost    │
└──────┬──────┘
       ▼
┌──────────────┐
│   Adım 5     │  Eşik karşılaştırma
│  Threshold   │──────────────────────────────────┐
└──────┬───────┘                                  │
       │                                          │
  P ≤ 0.35 veya P ≥ 0.65                 0.35 < P < 0.65
  (Kesin karar)                           (Belirsiz)
       │                                          │
       │                                          ▼
       │                                 ┌─────────────┐
       │                                 │  Adım 6     │  NLP temizleme + tokenization
       │                                 │  NLP        │
       │                                 └──────┬──────┘
       │                                        ▼
       │                                 ┌─────────────┐
       │                                 │  Adım 7     │  BiLSTM + GloVe + Attention → P₂
       │                                 │  BiLSTM     │
       │                                 └──────┬──────┘
       │                                        ▼
       │                                 ┌─────────────┐
       │                                 │  Adım 8     │  Stacking: 0.4×P₁ + 0.6×P₂
       │                                 │  Meta-Clf   │
       │                                 └──────┬──────┘
       │                                        │
       └────────────────┬───────────────────────┘
                        ▼
               ┌─────────────┐
               │   Adım 9    │  SAFE / UNCERTAIN / PHISHING + Eylem
               │   Sonuç     │
               └─────────────┘
```

### 2.2 Dallanmalı Pipeline (Hızlı Yol / Derin Analiz)

Sistem her e-postayı aynı yoldan geçirmez. Adaptif eşik mekanizması sayesinde iki farklı yol vardır:

| Yol | Koşul | Adımlar | Ortalama Süre |
|---|---|---|---|
| **Hızlı Yol** | P ≤ 0.35 veya P ≥ 0.65 | Adım 1→2→3→4→5→9 | ~50ms |
| **Derin Analiz** | 0.35 < P < 0.65 | Adım 1→2→3→4→5→6→7→8→9 | ~200ms |

- **Hızlı Yol (FAST_PATH):** XGBoost'un güvenle karar verebildiği e-postalar. Gereksiz yere BiLSTM çalıştırılmaz.
- **Derin Analiz (FULL_PIPELINE):** XGBoost'un emin olamadığı belirsiz vakalar. BiLSTM + Meta-Classifier devreye girer.

### 2.3 Teknoloji Yığını

| Katman | Teknoloji | Versiyon |
|---|---|---|
| **Dil** | Python | 3.12 |
| **Backend** | FastAPI + Uvicorn | 0.115 |
| **ML** | XGBoost | 2.1 |
| **DL** | TensorFlow / Keras | 2.19 |
| **Embedding** | GloVe 6B | 100d |
| **NLP** | NLTK + Regex | — |
| **Frontend** | HTML / CSS / JavaScript | Vanilla |
| **API Format** | REST JSON | v3.0.0 |

---

## 3. Veri Setleri

### 3.1 Kullanılan Veri Kaynakları

Sistem 6 farklı akademik ve açık kaynak veri setinden derlenen e-postalarla eğitilmiştir:

| # | Veri Seti | Toplam | Phishing | Safe | Phishing % | Kaynak |
|---|---|---|---|---|---|---|
| 1 | **CEAS_08** | 37,000 | 18,000 | 19,000 | %48.6 | CEAS Spam Challenge 2008 |
| 2 | **Enron** | 33,702 | 6,552 | 27,150 | %19.4 | Enron kurumsal e-postaları |
| 3 | **SpamAssassin** | 6,047 | 2,399 | 3,648 | %39.7 | Apache SpamAssassin |
| 4 | **Ling** | 2,893 | 481 | 2,412 | %16.6 | Dilbilim e-posta listesi |
| 5 | **Nazario** | 1,565 | 1,565 | 0 | %100.0 | Saf phishing corpus |
| 6 | **Nigerian Fraud** | 2,000 | 2,000 | 0 | %100.0 | 419 dolandırıcılık |
| | **Toplam** | **~83,207** | **~30,997** | **~52,210** | | |

### 3.2 Veri Ön İşleme (Preprocess)

Ham veri setlerinden nihai eğitim verisine dönüşüm süreci:

```
6 Ham CSV → Birleştirme → Duplikat Temizleme → Dengeli Örnekleme → final_dataset.csv
```

**Adımlar:**

| Adım | İşlem | Detay |
|---|---|---|
| 1 | **Kolon standardizasyonu** | Her veri setinden yalnızca `body`, `label`, `source` kolonları alınır |
| 2 | **Birleştirme** | `pd.concat()` ile tek DataFrame'e dönüştürülür |
| 3 | **Duplikat kaldırma** | MD5 hash ile birebir aynı e-postalar çıkarılır |
| 4 | **Dengeli örnekleme** | Her kaynaktan eşit sayıda örnek, sınıf dengesini koruyarak (stratified) seçilir |
| 5 | **Hedef boyut** | 18,650 e-posta (9,325 phishing + 9,325 safe) |

> Dosya: `phishing_project/preprocess.py`

### 3.3 Veri Bölme Stratejisi (Train / Val / Test)

```
final_dataset.csv (18,650 e-posta)
        │
        ├── %70 Train  → 13,055 örnek
        ├── %15 Val    →  2,797 örnek
        └── %15 Test   →  2,798 örnek
```

- **Stratified Split** kullanılır (`stratify=y`) — her bölümde phishing/safe oranı korunur.
- `random_state=42` ile tekrarlanabilir sonuçlar garanti edilir.
- Scaler yalnızca **Train** üzerinde `fit` edilir, Val ve Test'e yalnızca `transform` uygulanır (data leakage önlenir).

> Dosya: `phishing_project/split_data.py`

---

## 4. Özellik Çıkarma (Feature Extraction)

### 4.1 Stilometrik Özellikler (13 Feature)

E-posta metninin **yazım tarzını** (stilometri) sayısal değerlere dönüştüren 13 özellik çıkarılır:

| # | Özellik | Açıklama | Örnek |
|---|---|---|---|
| 1 | `num_chars` | Toplam karakter sayısı | 1,245 |
| 2 | `num_words` | Toplam kelime sayısı | 210 |
| 3 | `num_sentences` | Toplam cümle sayısı | 15 |
| 4 | `num_upper_chars` | Büyük harf sayısı | 38 |
| 5 | `num_special_chars` | Özel karakter sayısı (noktalama hariç) | 12 |
| 6 | `avg_word_size` | Ortalama kelime uzunluğu | 5.93 |
| 7 | `words_to_chars` | Kelime / karakter oranı | 0.168 |
| 8 | `special_chars_to_chars` | Özel karakter / karakter oranı | 0.009 |
| 9 | `unique_words_to_word` | Benzersiz kelime / toplam kelime | 0.714 |
| 10 | `exclamationmark_to_chars` | Ünlem / karakter oranı | 0.004 |
| 11 | `questionmark_to_chars` | Soru işareti / karakter oranı | 0.001 |
| 12 | `avg_words_in_sentence` | Cümle başına ortalama kelime | 14.0 |
| 13 | `max_words_in_sentence` | En uzun cümledeki kelime sayısı | 42 |

> Bu özellikler phishing e-postalarının tipik kalıplarını yakalar: kısa cümleler, aşırı ünlem, düşük kelime çeşitliliği, vb.

### 4.2 Özellik Ölçekleme (StandardScaler)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Sadece train'de fit
X_val_scaled   = scaler.transform(X_val)          # Transform only
X_test_scaled  = scaler.transform(X_test)         # Transform only
```

- Her özellik **μ=0, σ=1** olacak şekilde normalize edilir.
- Scaler `data/scaler.pkl` olarak kaydedilir ve runtime'da backend tarafından kullanılır.

> Dosya: `phishing_project/feature_extraction.py`, `split_data.py`

---

## 5. Model Mimarisi

### 5.1 XGBoost Classifier (Branch 1)

Stilometrik özelliklerden (13 feature) phishing olasılığı tahmin eder.

| Parametre | Değer |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `eval_metric` | logloss |
| `early_stopping_rounds` | 20 |

**Girdi:** 13 stilometrik feature (scaled)
**Çıktı:** P₁ ∈ [0.0, 1.0] — Phishing olasılığı

> Dosya: `phishing_project/train_xgb.py`

### 5.2 BiLSTM + GloVe + Attention (Branch 2)

E-posta metninin anlamsal içeriğinden phishing olasılığı tahmin eder.

```
Padded Sequence [batch, 200]
          │
    ┌─────┴──────┐
    │             │
    ▼             ▼
[GloVe Branch]  [BiLSTM Branch]
  Embed(100d)     Embed(100d, trainable)
  frozen            │
    │             BiLSTM(128 units)
  GlobalAvg       + Self-Attention
  Pooling           │
    │             [batch, 256]
  [batch, 100]      │
    │               │
    └─────┬─────────┘
          ▼
    Concatenate [batch, 356]
          │
    Dense(128) → Dropout(0.4)
    Dense(64)  → Dropout(0.3)
    Dense(1, sigmoid)
```

#### 5.2.1 GloVe Embedding (Frozen)

- **Kaynak:** GloVe 6B, 100 boyutlu (Stanford NLP)
- **Kelime:** 400,000 kelime × 100d vektör
- **Kapsama:** Vocabulary'deki kelimelerin ~%85'ini karşılar
- **Trainable:** Hayır (frozen) — önceden öğrenilmiş anlamsal bilgi korunur
- Çıktı: `GlobalAveragePooling1D` → `[batch, 100]`

#### 5.2.2 BiLSTM Katmanı (Trainable)

- **Yön:** Çift yönlü (Bidirectional) — hem ileri hem geri bağlam
- **Birimler:** 128 (×2 = 256 çıktı boyutu)
- **return_sequences:** True (Attention için tüm zaman adımları)
- **Dropout:** 0.2, Recurrent Dropout: 0.2
- **Trainable Embedding:** 100d — eğitim sırasında optimize edilir

#### 5.2.3 Self-Attention Mekanizması

Bahdanau-style dikkat mekanizması:

```
lstm_out [batch, 200, 256]
    │
    ▼ Dense(1, tanh)
attention_scores [batch, 200, 1]
    │
    ▼ Flatten → Softmax
attention_weights [batch, 200]
    │
    ▼ RepeatVector(256) → Permute
    ▼ Multiply(lstm_out × weights)
    ▼ Sum(axis=1)
context_vector [batch, 256]
```

Bu mekanizma modelin **hangi kelimelere odaklanması gerektiğini** öğrenmesini sağlar (ör: "verify", "suspended", "urgent").

#### 5.2.4 Fusion (Birleştirme)

```python
merged = Concatenate()([glove_pool, lstm_context])  # [batch, 356]
```

- GloVe (100d): Genel anlamsal temsil
- BiLSTM + Attention (256d): Bağlam-duyarlı özellikler
- Toplam: 356 boyutlu zengin temsil → Dense katmanlarla sınıflandırma

### 5.3 Meta-Classifier (Stacking)

İki modelin çıktılarını birleştiren Level-2 karar mekanizması:

**Meta-Feature'lar:**

| Feature | Formül | Açıklama |
|---|---|---|
| P₁ | XGBoost çıktısı | Stilometrik skor |
| P₂ | BiLSTM çıktısı | İçerik skoru |
| P₁ × P₂ | Çarpım | Çarpımsal uyum |
| \|P₁ − P₂\| | Fark | Uyumsuzluk ölçüsü |
| (P₁ + P₂) / 2 | Ortalama | Merkezi tahmin |

**Ağırlıklı birleştirme:**

```
P_final = 0.4 × P₁(XGBoost) + 0.6 × P₂(BiLSTM)
```

**Uyumsuzluk yönetimi:**

| Durum | Koşul | Strateji |
|---|---|---|
| HIGH agreement | \|P₁−P₂\| ≤ 0.15 | Ağırlıklı ortalama |
| MEDIUM agreement | 0.15 < \|P₁−P₂\| ≤ 0.30 | Ağırlıklı ortalama |
| LOW agreement | \|P₁−P₂\| > 0.30 | Basit ortalama (güvenli yol) |

> Dosya: `backend/app/agents/agent8_meta_classifier.py`

### 5.4 Adaptif Eşik Mekanizması

Sistem üç sınıf kararı verir:

| Olasılık Aralığı | Karar | Eylem |
|---|---|---|
| P ≥ 0.80 | PHISHING | 🔴 QUARANTINE |
| 0.65 ≤ P < 0.80 | PHISHING | 🟠 SPAM_FOLDER |
| 0.35 < P < 0.65 | UNCERTAIN | 🟡 MANUAL_REVIEW |
| P ≤ 0.35 | SAFE | 🟢 INBOX |

Eşik değerleri (`T_low=0.35`, `T_high=0.65`) runtime'da frontend üzerinden **Monitor** sayfasında değiştirilebilir.

---

## 6. NLP Pipeline

### 6.1 Metin Temizleme Adımları

E-posta metni BiLSTM modeline girmeden önce aşağıdaki temizleme adımlarından geçer:

| # | Adım | Regex / İşlem | Örnek |
|---|---|---|---|
| 1 | **Lowercase** | `text.lower()` | "URGENT" → "urgent" |
| 2 | **URL kaldırma** | `re.sub(r'http\S+', '')` | "http://evil.com" → "" |
| 3 | **E-posta kaldırma** | `re.sub(r'\S+@\S+', '')` | "user@bank.com" → "" |
| 4 | **HTML tag kaldırma** | `re.sub(r'<[^>]+>', '')` | "\<div\>text\</div\>" → "text" |
| 5 | **Özel karakter filtre** | `re.sub(r'[^a-z\s]', '')` | "hello!!!" → "hello" |
| 6 | **Boşluk normalize** | `re.sub(r'\s+', ' ')` | "hello   world" → "hello world" |

> Dosya: `backend/app/agents/agent2_preprocessing.py`

### 6.2 Tokenization

```python
tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
```

| Parametre | Değer |
|---|---|
| Vocabulary boyutu | 50,000 kelime |
| OOV token | `<OOV>` (bilinmeyen kelimeler için) |
| Kaydedilen dosya | `LSTM/tokenizer.pkl` (~20.5 MB) |

### 6.3 Padding (maxlen=200)

```python
padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
```

- **maxlen=200:** Her e-posta 200 token uzunluğuna sabitlenir
- Kısa metinler → **sıfır doldurma** (post-padding)
- Uzun metinler → **kırpma** (post-truncating)

---

## 7. 9 Adımlı İşlem Pipeline'ı

### 7.1 Adım 1 — Ingestion

Kullanıcıdan gelen e-posta verilerini alır ve birleştirir.

```python
combined_text = f"{subject} {body}"
```

- **Girdi:** `sender` (opsiyonel), `subject`, `body`
- **Çıktı:** Birleştirilmiş metin
- **Dosya:** `agent1_ingestion.py`

### 7.2 Adım 2 — Paralel Ön İşleme

İki işlemi `asyncio.gather()` ile eşzamanlı çalıştırır:

| Thread | İşlem | Çıktı |
|---|---|---|
| **A** | Metin temizleme (lowercase, regex) | `cleaned_text` |
| **B** | SHA-256 hash hesaplama | `sha256_body`, `sha256_subject` |

Ayrıca **duplikat tespiti** yapar — aynı e-posta daha önce analiz edilmişse runtime cache'den tespit eder.

- **Dosya:** `agent2_preprocessing.py`

### 7.3 Adım 3 — Stilometrik Extraction

Temizlenmiş metinden 13 stilometrik özellik çıkarır (bkz. [Bölüm 4.1](#41-stilometrik-özellikler-13-feature)).

- **Girdi:** Ham metin (temizlenmemiş — stilometri ham metin üzerinden hesaplanır)
- **Çıktı:** 13 boyutlu özellik vektörü
- **Dosya:** `agent3_stylometric.py`

### 7.4 Adım 4 — XGBoost Tahmin

StandardScaler ile ölçeklenmiş 13 özelliği XGBoost modeline vererek P₁ olasılığını hesaplar.

```
[13 features] → StandardScaler → XGBoost → P₁ ∈ [0.0, 1.0]
```

- **Dosya:** `agent4_xgboost.py`

### 7.5 Adım 5 — Threshold (Eşik Karar)

P₁ değerini adaptif eşiklerle karşılaştırarak pipeline'ın dallanmasına karar verir.

```
P₁ ≤ T_low (0.35)   → SAFE       → Hızlı yol (Adım 9'a atla)
P₁ ≥ T_high (0.65)  → PHISHING   → Hızlı yol (Adım 9'a atla)
T_low < P₁ < T_high → UNCERTAIN  → Derin analiz (Adım 6'ya devam)
```

- **Dosya:** `agent5_threshold.py`

### 7.6 Adım 6 — NLP Ön İşleme

Yalnızca UNCERTAIN vakalar için çalışır. Metni tokenize ve padding uygular.

```
cleaned_text → Tokenizer(50K) → pad_sequences(maxlen=200) → [1, 200] array
```

- **Dosya:** `agent6_nlp.py`

### 7.7 Adım 7 — BiLSTM + GloVe + Attention

Padded sequence'ı BiLSTM modeline vererek P₂ olasılığını hesaplar (bkz. [Bölüm 5.2](#52-bilstm--glove--attention-branch-2)).

```
[1, 200] → BiLSTM_GloVe_Attention → P₂ ∈ [0.0, 1.0]
```

- **Model:** 10,288,970 parametre
- **Dosya:** `agent7_dual_branch.py`

### 7.8 Adım 8 — Meta-Classifier

P₁ (XGBoost) ve P₂ (BiLSTM) çıktılarını stacking ile birleştirir (bkz. [Bölüm 5.3](#53-meta-classifier-stacking)).

```
P_final = 0.4 × P₁ + 0.6 × P₂   (eğer agreement HIGH/MEDIUM ise)
P_final = (P₁ + P₂) / 2          (eğer agreement LOW ise)
```

- **Dosya:** `agent8_meta_classifier.py`

### 7.9 Adım 9 — Sonuç Üretimi

Final olasılığı eylem eşlemeleri ile birleştirerek kullanıcıya sonuç döner.

**Çıktı formatı:**

| Alan | Açıklama |
|---|---|
| `verdict` | SAFE / UNCERTAIN / PHISHING |
| `final_probability` | 0.0 – 1.0 |
| `action` | INBOX / MANUAL_REVIEW / SPAM_FOLDER / QUARANTINE |
| `detection_path` | FAST_PATH veya FULL_PIPELINE |
| `processing_time_ms` | Toplam işlem süresi |
| `explanation` | İnsan-okunur açıklama |

- **Dosya:** `agent9_output.py`

---

## 8. Backend (FastAPI)

### 8.1 API Endpoint'leri

| Metod | Endpoint | Açıklama |
|---|---|---|
| `GET` | `/api/v1/health` | Sistem sağlığı (model durumu, eşikler, versiyon) |
| `POST` | `/api/v1/predict` | Tek e-posta analizi |
| `POST` | `/api/v1/predict/batch` | Toplu analiz (max 100 e-posta) |
| `POST` | `/api/v1/evaluate` | CSV yükle → 10 performans metriği |
| `PATCH` | `/api/v1/threshold` | Adaptif eşik değerlerini güncelle |
| `GET` | `/docs` | Swagger UI (otomatik API dokümantasyonu) |
| `GET` | `/redoc` | ReDoc (alternatif dokümantasyon) |

**Örnek — Predict request:**

```json
POST /api/v1/predict
{
    "subject": "URGENT: Verify your account immediately",
    "body": "Dear Customer, Your account will be suspended. Click here: http://paypa1.com/verify",
    "sender": "security@paypa1.com"
}
```

**Örnek — Predict response:**

```json
{
    "verdict": "PHISHING",
    "final_probability": 0.8742,
    "xgboost_score": 0.7891,
    "lstm_score": 0.9214,
    "detection_path": "FULL_PIPELINE",
    "action": "QUARANTINE",
    "processing_time_ms": 187.4,
    "explanation": "Bu e-posta yüksek phishing riski taşıyor..."
}
```

### 8.2 Proje Yapısı

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              ← FastAPI app + middleware
│   ├── config.py            ← Tüm konfigürasyon sabitleri
│   ├── agents/
│   │   ├── agent1_ingestion.py         ← E-posta alımı
│   │   ├── agent2_preprocessing.py     ← Paralel temizleme + SHA-256
│   │   ├── agent3_stylometric.py       ← 13 stilometrik feature
│   │   ├── agent4_xgboost.py           ← XGBoost tahmin
│   │   ├── agent5_threshold.py         ← Adaptif eşik kararı
│   │   ├── agent6_nlp.py              ← Tokenization + padding
│   │   ├── agent7_dual_branch.py       ← BiLSTM + GloVe + Attention
│   │   ├── agent8_meta_classifier.py   ← Stacking meta-sınıflandırıcı
│   │   └── agent9_output.py            ← Sonuç formatlama + loglama
│   ├── routers/
│   │   ├── predict.py       ← /predict, /predict/batch
│   │   ├── evaluate.py      ← /evaluate (CSV upload)
│   │   └── health.py        ← /health, /threshold
│   ├── schemas/
│   │   ├── request.py       ← Pydantic request modelleri
│   │   └── response.py      ← Pydantic response modelleri
│   └── utils/
│       ├── logger.py        ← Logging konfigürasyonu
│       └── metrics.py       ← Performans metrik hesaplama
└── requirements.txt
```

### 8.3 Konfigürasyon (config.py)

| Sabit | Değer | Açıklama |
|---|---|---|
| `XGBOOST_MODEL_PATH` | `phishing_project/data/xgboost_model.pkl` | XGBoost model dosyası |
| `LSTM_MODEL_PATH` | `LSTM/bilstm_model.keras` | BiLSTM model dosyası |
| `TOKENIZER_PATH` | `LSTM/tokenizer.pkl` | Tokenizer dosyası |
| `SCALER_PATH` | `phishing_project/data/scaler.pkl` | StandardScaler dosyası |
| `THRESHOLD_LOW` | 0.35 | Safe sınırı |
| `THRESHOLD_HIGH` | 0.65 | Phishing sınırı |
| `LSTM_MAX_SEQUENCE_LENGTH` | 200 | Token padding uzunluğu |
| `API_VERSION` | "3.0.0" | API versiyonu |

### 8.4 Model Yükleme ve Servis

- **XGBoost:** `joblib.load()` ile yüklenir, ilk istekte lazy-load
- **BiLSTM:** Model mimarisi kodda rebuild edilir, `.keras` dosyasından weights yüklenir (Keras 3 Lambda serileştirme sorunu nedeniyle)
- **Tokenizer:** `pickle.load()` ile yüklenir
- **Scaler:** `joblib.load()` ile yüklenir
- Tüm modeller **singleton pattern** ile bir kez yüklenir, sonraki isteklerde cache'ten kullanılır

---

## 9. Frontend (Dashboard)

Dört sekmeli tek sayfalık uygulama (SPA):

### 9.1 Analiz Sayfası

Ana e-posta analiz arayüzü:

- **Gönderen** (opsiyonel), **Konu** ve **İçerik** alanları
- "Analiz Et" butonu → Backend `/predict` endpoint'ine istek
- **Sonuç banner'ı:** SAFE (yeşil), UNCERTAIN (sarı), PHISHING (kırmızı)
- **Metrik kartları:** Güven Skoru, XGBoost, BiLSTM, Süre, Path, Eylem
- **Pipeline animasyonu:** 9 adımlı görsel pipeline (FAST_PATH vs FULL_PIPELINE)
- **Açıklama kutusu:** İnsan-okunur sonuç açıklaması
- **SHAP Feature Importance:** Stilometrik özelliklerin radar grafiği

### 9.2 Değerlendirme Sayfası

#### 9.2.1 Performans Metrikleri

CSV yükleme ile model performansını ölçen 10 metrik kartı:

| Metrik | Açıklama |
|---|---|
| Accuracy | Genel doğruluk |
| F1-Score | Precision + Recall dengesi |
| ROC-AUC | Sınıflandırma kalitesi |
| PR-AUC | Precision-Recall eğri altı alan |
| Precision | Phishing tahmin hassasiyeti |
| Recall | Phishing yakalama oranı |
| MCC | Matthews korelasyon katsayısı |
| FNR | Kaçan phishing oranı |
| Avg Latency | Ortalama işlem süresi |
| Throughput | Saniyedeki istek sayısı |

#### 9.2.2 Veri Seti Analiz Havuzu

- 6 temel veri setinin kartları (istatistik + phishing % barı)
- "Yeni Veri Seti Ekle" butonu ile CSV yükleme
- Otomatik `label` sütunu parse (phishing/safe sayısı)
- Dinamik kart oluşturma + kaldırma

### 9.3 Monitor Sayfası

#### 9.3.1 Adaptif Eşik Kontrolü

- T_low ve T_high slider'ları ile eşik değerlerini ayarlama
- "Kaydet" butonu → Backend `/threshold` PATCH endpoint'i

#### 9.3.2 Sistem Mimarisi (Dallanmalı Diyagram)

- Adım 1–5 üst sıra (her zaman çalışır)
- Çatallanma: SAFE/PHISHING → Hızlı Yol, UNCERTAIN → Derin Analiz (6→7→8)
- Birleşme: Adım 9 (Sonuç)

#### 9.3.3 Sistem Sağlığı

- Backend bağlantı durumu (🟢 Bağlı / 🔴 Bağlantı yok)
- Model yükleme durumları: XGBoost ✅, BiLSTM ✅, Tokenizer ✅
- Versiyon ve eşik bilgileri

### 9.4 Nasıl Çalışır Sayfası

Sistemin çalışma mantığını adım adım açıklayan interaktif dokümantasyon:

- Genel mimari açıklaması
- NLP ön işleme adımları tablosu
- Adaptif eşik mekanizması açıklaması
- BiLSTM dual-branch mimari diyagramı
- Stacking meta-classifier açıklaması
- 9 adımlı pipeline kartları

---

## 10. Model Eğitimi

### 10.1 XGBoost Eğitimi

```bash
cd phishing_project
python train_xgb.py
```

**Pipeline:** `featured_dataset.csv` → `split_data.py` (bölme + ölçekleme) → `train_xgb.py`

- **Girdi:** 13 stilometrik feature (scaled)
- **Early Stopping:** 20 round (val logloss)
- **Çıktı:** `data/xgboost_model.pkl`

### 10.2 BiLSTM Eğitimi

```bash
cd phishing_project
python download_glove.py    # GloVe 6B 100d indir (ilk seferde)
python train_bilstm.py      # BiLSTM eğitimi
```

**Pipeline:** `final_dataset.csv` → Tokenization → GloVe Matrix → BiLSTM eğitim

- **Girdi:** E-posta metinleri (padded, maxlen=200)
- **GloVe:** 6B, 100d — 400K kelime
- **EarlyStopping:** `val_auc` monitor, patience=5, restore_best_weights
- **ReduceLROnPlateau:** `val_loss` monitor, factor=0.5, patience=3
- **Çıktı:** `LSTM/bilstm_model.keras`, `LSTM/tokenizer.pkl`

### 10.3 Hiperparametre Tablosu

| Parametre | XGBoost | BiLSTM |
|---|---|---|
| Epochs / Estimators | — / 300 | 15 / — |
| Batch Size | — | 64 |
| Learning Rate | 0.1 | 1e-3 (Adam) |
| Max Depth / Units | 6 / — | — / 128 |
| Dropout | — | 0.4 / 0.3 |
| Embedding Dim | — | 100 (GloVe) |
| Vocab Size | — | 50,000 |
| Max Sequence Length | — | 200 |
| Early Stopping | 20 rounds | 5 epoch (val_auc) |
| Random State | 42 | 42 |

### 10.4 Eğitim Sonuçları

**BiLSTM Eğitim İlerlemesi:**

| Epoch | Train Acc | Train AUC | Val Acc | Val AUC | Durum |
|---|---|---|---|---|---|
| 1 | 0.7788 | 0.8560 | 0.9557 | **0.9944** | ✅ Best |
| 2 | 0.9737 | 0.9951 | 0.9653 | 0.9887 | — |
| 3 | 0.9908 | 0.9986 | 0.9617 | 0.9886 | — |
| 4 | 0.9951 | 0.9996 | 0.9632 | 0.9861 | — |
| 5 | 0.9971 | 0.9998 | 0.9693 | 0.9830 | LR ↓ |
| 6 | 0.9983 | 0.9996 | 0.9713 | 0.9882 | EarlyStopping |

> EarlyStopping en iyi epoch 1'in ağırlıklarını geri yükledi (val_auc=0.9944).

---

## 11. Performans Metrikleri

### 11.1 XGBoost Sonuçları

| Metrik | Değer |
|---|---|
| Accuracy | 0.9232 |
| Precision | 0.9180 |
| Recall | 0.9291 |
| F1-Score | 0.9235 |

### 11.2 BiLSTM Sonuçları

| Metrik | Değer |
|---|---|
| Accuracy | **0.9693** |
| Precision | **0.9626** |
| Recall | **0.9764** |
| F1-Score | **0.9695** |
| ROC-AUC | **0.9944** |

**Confusion Matrix (BiLSTM — Test Seti, 2798 örnek):**

```
              Predicted
            Safe    Phishing
Actual Safe   1346      53     (FP rate: 3.8%)
     Phish     33    1366     (FN rate: 2.4%)
```

### 11.3 LSTM vs BiLSTM Karşılaştırması

| Özellik | Eski LSTM | **Yeni BiLSTM** | Değişim |
|---|---|---|---|
| **Accuracy** | 0.9564 | **0.9693** | +1.29% |
| **F1-Score** | — | **0.9695** | — |
| **ROC-AUC** | — | **0.9944** | — |
| maxlen | 100 | **200** | 2× bağlam |
| Embedding | GloVe tek dal | **GloVe + BiLSTM dual** | Çift dal |
| Attention | Yok | **Self-Attention** | Eklendi |
| Yön | Tek yönlü → | **Çift yönlü ↔** | BiLSTM |

---

## 12. Kurulum ve Çalıştırma

### 12.1 Gereksinimler

| Gereksinim | Minimum |
|---|---|
| Python | 3.12+ |
| RAM | 4 GB |
| Disk | ~1 GB (modeller + veri) |
| İşletim Sistemi | Windows 10/11, Linux, macOS |

### 12.2 Kurulum Adımları

```bash
# 1. Sanal ortam oluştur
python -m venv venv_phish
venv_phish\Scripts\activate   # Windows

# 2. Bağımlılıkları yükle
pip install -r backend/requirements.txt

# 3. NLTK verilerini indir
python -c "import nltk; nltk.download('stopwords')"

# 4. GloVe indir (sadece eğitim için, ~330 MB)
cd phishing_project
python download_glove.py
```

### 12.3 Tek Komutla Başlatma

```bash
# Windows — start.bat çalıştır:
start.bat

# Manuel başlatma:
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (ayrı terminal):
cd frontend
python -m http.server 8090
```

**Erişim:**
- Frontend: `http://localhost:8090`
- API Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/v1/health`

---

## 13. Proje Dosya Yapısı

```
Ağ ödevi/
├── backend/                        ← FastAPI backend
│   ├── app/
│   │   ├── agents/                 ← 9 pipeline adımı
│   │   │   ├── agent1_ingestion.py
│   │   │   ├── agent2_preprocessing.py
│   │   │   ├── agent3_stylometric.py
│   │   │   ├── agent4_xgboost.py
│   │   │   ├── agent5_threshold.py
│   │   │   ├── agent6_nlp.py
│   │   │   ├── agent7_dual_branch.py
│   │   │   ├── agent8_meta_classifier.py
│   │   │   └── agent9_output.py
│   │   ├── routers/                ← API endpoint'leri
│   │   │   ├── predict.py
│   │   │   ├── evaluate.py
│   │   │   └── health.py
│   │   ├── schemas/                ← Pydantic modelleri
│   │   ├── utils/                  ← Yardımcı fonksiyonlar
│   │   ├── config.py               ← Konfigürasyon sabitleri
│   │   └── main.py                 ← FastAPI app
│   └── requirements.txt
│
├── frontend/                       ← Dashboard
│   ├── index.html                  ← Ana sayfa (SPA)
│   ├── charts.js                   ← Grafik bileşenleri
│   └── howitworks.js               ← "Nasıl Çalışır" içeriği
│
├── LSTM/                           ← Model dosyaları
│   ├── bilstm_model.keras          ← BiLSTM model (~83 MB)
│   └── tokenizer.pkl               ← Tokenizer (~20 MB)
│
├── phishing_project/               ← Eğitim scriptleri + veri
│   ├── data/
│   │   ├── CEAS_08.csv             ← Ham veri setleri
│   │   ├── Enron.csv
│   │   ├── SpamAssasin.csv
│   │   ├── Ling.csv
│   │   ├── Nazario.csv
│   │   ├── Nigerian_Fraud.csv
│   │   ├── final_dataset.csv       ← Birleştirilmiş eğitim verisi
│   │   ├── featured_dataset.csv    ← Stilometrik feature'lı veri
│   │   ├── glove.6B.100d.txt       ← GloVe embedding (~347 MB)
│   │   ├── xgboost_model.pkl       ← Eğitilmiş XGBoost
│   │   └── scaler.pkl              ← StandardScaler
│   ├── preprocess.py               ← Veri birleştirme + dengeleme
│   ├── feature_extraction.py       ← Stilometrik özellik çıkarma
│   ├── split_data.py               ← Train/Val/Test bölme
│   ├── train_xgb.py                ← XGBoost eğitimi
│   ├── train_bilstm.py             ← BiLSTM eğitimi
│   └── download_glove.py           ← GloVe indirici
│
├── start.bat                       ← Tek tıkla başlatma (Windows)
└── READMEfull.md                   ← Bu dosya
```

---

## 14. Kaynakça

| # | Kaynak | Açıklama |
|---|---|---|
| 1 | Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP 2014. | GloVe embedding yöntemi |
| 2 | Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016. | XGBoost algoritması |
| 3 | Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8). | LSTM mimarisi |
| 4 | Schuster, M., & Paliwal, K. K. (1997). *Bidirectional Recurrent Neural Networks*. IEEE Transactions on Signal Processing. | BiLSTM mimarisi |
| 5 | Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015. | Attention mekanizması |
| 6 | CEAS Spam Challenge 2008 Dataset | CEAS_08 veri seti |
| 7 | Enron Email Dataset (Klimt & Yang, 2004) | Enron veri seti |
| 8 | Apache SpamAssassin Public Corpus | SpamAssassin veri seti |
| 9 | Nazario Phishing Corpus | Nazario veri seti |
| 10 | Nigerian Fraud Email Dataset | Nigerian Fraud veri seti |

---

## 15. Geliştirici

<div align="center">

**seydieryilmazz**

[![GitHub](https://img.shields.io/badge/GitHub-seydivakkas-181717?style=for-the-badge&logo=github)](https://github.com/seydivakkas)

</div>

---

## 📄 Lisans

Bu proje **All Rights Reserved** lisansı altındadır. Kopyalanması, değiştirilmesi ve dağıtılması yasaktır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

---

<div align="center">

**PhishGuard AI v3.0** — BiLSTM + XGBoost Tabanlı Hibrit Phishing Tespit Sistemi

Made with ❤️ by [seydieryilmazz](https://github.com/seydivakkas)

</div>
