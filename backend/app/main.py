"""
Phishing Detection AI Agent System — FastAPI Backend
Stylometric + Stacking Tabanlı Adaptif Phishing Tespiti v3.0
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import time

from app.config import API_TITLE, API_VERSION, CORS_ORIGINS
from app.routers import predict, evaluate, health
from app.utils.logger import setup_logger

logger = setup_logger("phishing.api")

app = FastAPI(
    title=API_TITLE,
    description="Stylometric + Dual-Branch Stacking Tabanlı 9-Agent Phishing Tespit Sistemi",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — Frontend için
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# İstek süresi middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(duration, 2))
    logger.info(f"{request.method} {request.url.path} → {duration:.1f}ms")
    return response


# Router'ları bağla
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(evaluate.router, prefix="/api/v1", tags=["Evaluation"])


@app.get("/")
async def root():
    return {
        "system": "Phishing Detection AI Agent System",
        "version": API_VERSION,
        "agents": 9,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info(f"  [SYSTEM] {API_TITLE} v{API_VERSION}")
    logger.info("  [DOCS]   http://localhost:8000/docs")
    logger.info("=" * 60)
    # Model on yukleme
    try:
        from app.agents.agent4_xgboost import is_loaded
        if is_loaded():
            logger.info("  [OK] XGBoost + Scaler yuklendi")
    except Exception as e:
        logger.warning(f"  [WARN] XGBoost yuklenemedi: {e}")
    try:
        from app.agents.agent6_nlp import is_loaded
        if is_loaded():
            logger.info("  [OK] Tokenizer yuklendi")
    except Exception as e:
        logger.warning(f"  [WARN] Tokenizer yuklenemedi: {e}")
