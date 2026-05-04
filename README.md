<div align="center">

# 🛡️ PhishGuard AI

### Hibrit Derin Öğrenme ile E-posta Phishing Tespiti

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-189FDD?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=for-the-badge)](LICENSE)

<br>

**BiLSTM + GloVe + Self-Attention** ve **XGBoost** modellerini birleştiren,<br>
9 adımlı adaptif pipeline ile phishing e-postalarını tespit eden hibrit AI sistemi.

<br>

| Accuracy | F1-Score | ROC-AUC | Veri Seti |
|:---:|:---:|:---:|:---:|
| **96.93%** | **96.95%** | **0.9944** | **18,650 e-posta** |

</div>

---

## ⚡ Öne Çıkan Özellikler

<table>
<tr>
<td width="50%">

### 🧠 Dual-Branch Stacking
XGBoost (stilometrik analiz) + BiLSTM (anlamsal analiz) çıktılarını Meta-Classifier ile birleştirir.

### 🔀 Adaptif Pipeline
Kesin kararlar **hızlı yoldan** (~50ms), belirsiz vakalar **derin analizden** (~200ms) geçer.

### 📊 13 Stilometrik Özellik
Karakter dağılımı, kelime çeşitliliği, ünlem oranı gibi yazım tarzı kalıplarını yakalar.

</td>
<td width="50%">

### 🎯 Self-Attention Mekanizması
"verify", "suspended", "urgent" gibi phishing anahtar kelimelerine otomatik odaklanır.

### 🌐 GloVe Embedding (100d)
400K kelimelik önceden eğitilmiş vektörlerle derin anlamsal temsil.

### 📈 Gerçek Zamanlı Dashboard
Analiz, değerlendirme, sistem izleme ve mimari görselleştirme.

</td>
</tr>
</table>

---

## 🏗️ Sistem Mimarisi

```
E-posta Girişi
      │
  ┌───┴───┐
  │ Adım  │  1. Ingestion → 2. Preprocess → 3. Stilometrik
  │ 1-4   │  4. XGBoost Tahmin → P₁
  └───┬───┘
      │
  ┌───┴────┐
  │ Adım 5 │  Adaptif Eşik Kararı
  │Threshold│
  └┬──────┬┘
   │      │
   │   P belirsiz (0.35 < P < 0.65)
   │      │
   │  ┌───┴───┐
   │  │Deep   │  6. NLP → 7. BiLSTM+GloVe+Attention → P₂
   │  │Path   │  8. Meta-Classifier (0.4×P₁ + 0.6×P₂)
   │  └───┬───┘
   │      │
   └──┬───┘
      │
  ┌───┴───┐
  │ Adım 9│  ✅ SAFE → INBOX
  │ Sonuç │  ⚠️ UNCERTAIN → MANUAL_REVIEW
  └───────┘  🚨 PHISHING → QUARANTINE
```

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Klonla
git clone https://github.com/seydivakkas/PhishGuard-AI.git
cd PhishGuard-AI

# 2. Sanal ortam
python -m venv venv_phish
venv_phish\Scripts\activate        # Windows

# 3. Bağımlılıklar
pip install -r backend/requirements.txt

# 4. Başlat
start.bat                          # Windows (tek tık)

# veya manuel:
cd backend && python -m uvicorn app.main:app --port 8000
cd frontend && python -m http.server 8090
```

| Servis | URL |
|---|---|
| **Dashboard** | http://localhost:8090 |
| **API Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/api/v1/health |

---

## 📊 Performans

<table>
<tr>
<td>

### BiLSTM + GloVe + Attention

| Metrik | Değer |
|---|---|
| Accuracy | **96.93%** |
| Precision | **96.26%** |
| Recall | **97.64%** |
| F1-Score | **96.95%** |
| ROC-AUC | **0.9944** |

</td>
<td>

### Confusion Matrix (Test: 2798)

```
              Predicted
            Safe    Phishing
Actual Safe  1346      53
     Phish    33     1366
```

- **FP Rate:** 3.8%
- **FN Rate:** 2.4% ← Phishing kaçırma oranı çok düşük

</td>
</tr>
</table>

### LSTM → BiLSTM Geçişi

| Metrik | Eski LSTM | BiLSTM | Değişim |
|---|---|---|---|
| Accuracy | 95.64% | **96.93%** | +1.29% |
| Attention | Yok | ✅ Self-Attention | Eklendi |
| Yön | Tek yönlü → | ↔ Çift yönlü | BiLSTM |
| Bağlam | 100 token | **200 token** | 2× |

---

## 🗂️ Proje Yapısı

```
PhishGuard-AI/
├── backend/                    # FastAPI backend
│   └── app/
│       ├── agents/             # 9 pipeline adımı (agent1-9)
│       ├── routers/            # predict, evaluate, health
│       ├── schemas/            # Pydantic modelleri
│       └── config.py           # Konfigürasyon
├── frontend/                   # Cyberpunk dashboard
│   ├── index.html              # Ana SPA
│   ├── charts.js               # Grafik bileşenleri
│   └── howitworks.js           # Pipeline açıklamaları
├── LSTM/                       # Model dosyaları
│   ├── bilstm_model.keras      # BiLSTM model (~83 MB)
│   └── tokenizer.pkl           # Tokenizer (~20 MB)
├── phishing_project/           # Eğitim scriptleri
│   ├── data/                   # Veri setleri + modeller
│   ├── train_bilstm.py         # BiLSTM eğitimi
│   └── train_xgb.py            # XGBoost eğitimi
├── READMEfull.md               # 📖 Detaylı dokümantasyon
├── start.bat                   # Tek tıkla başlatma
└── LICENSE                     # MIT
```

---

## 📖 Detaylı Dokümantasyon

Sistemin tüm teknik detayları için **[READMEfull.md](READMEfull.md)** dosyasına göz atın:

- 14 bölüm, kapsamlı teknik açıklama
- Model mimarisi diyagramları
- Veri seti analizleri
- API endpoint dokümantasyonu
- Hiperparametre tabloları
- Eğitim sonuçları ve karşılaştırmalar

---

## 🔧 Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| **Backend** | FastAPI + Uvicorn |
| **ML** | XGBoost (Stilometrik) |
| **DL** | TensorFlow / Keras (BiLSTM) |
| **Embedding** | GloVe 6B 100d |
| **NLP** | NLTK + Regex |
| **Frontend** | HTML / CSS / JS |
| **Veri** | 6 akademik veri seti |

---

## 📄 Lisans

Bu proje **All Rights Reserved** lisansı altındadır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

---

<div align="center">

**PhishGuard AI v3.0**

Geliştirici: **[seydieryilmazz](https://github.com/seydivakkas)**

Made with ❤️

</div>
