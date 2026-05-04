"""
Evaluate Router — Model performans değerlendirmesi.
POST /evaluate → CSV upload → 10 metrik
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.response import PerformanceMetrics
from app.schemas.request import EmailRequest
from app.routers.predict import _run_pipeline
from app.utils.metrics import compute_metrics
import pandas as pd
import io
import time

router = APIRouter()


@router.post("/evaluate", response_model=PerformanceMetrics)
async def evaluate_model(file: UploadFile = File(...)):
    """
    CSV yükle ve model performansını değerlendir.
    CSV sütunları: subject, body, label (0=safe, 1=phishing)
    """
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"CSV okuma hatası: {e}")

    required = {"body", "subject", "label"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(400, f"Eksik sütunlar: {missing}")

    predictions = []
    latencies = []

    for _, row in df.iterrows():
        t0 = time.time()
        try:
            result = await _run_pipeline(EmailRequest(
                subject=str(row["subject"]),
                body=str(row["body"])
            ))
            predictions.append(1 if result["verdict"] == "PHISHING" else 0)
        except Exception:
            predictions.append(0)
        latencies.append((time.time() - t0) * 1000)

    metrics = compute_metrics(df["label"].tolist(), predictions, latencies)
    return PerformanceMetrics(**metrics)
