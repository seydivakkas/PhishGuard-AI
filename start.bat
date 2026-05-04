@echo off
chcp 65001 >nul
echo.
echo  ╔══════════════════════════════════════╗
echo  ║   PhishGuard AI Agent System v3.0   ║
echo  ║   BiLSTM + XGBoost Aktif             ║
echo  ╚══════════════════════════════════════╝
echo.
cd /d "%~dp0backend"

echo  [INFO] venv_phish aktiflestirildi (Python 3.12 + TensorFlow 2.21)
echo  [INFO] API Docs   : http://localhost:8000/docs
echo  [INFO] Frontend   : http://localhost:8080 (ayri terminalde python -m http.server 8080)
echo  [INFO] Saglik     : http://localhost:8000/api/v1/health
echo.

set KERAS_BACKEND=tensorflow
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2

..\venv_phish\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
