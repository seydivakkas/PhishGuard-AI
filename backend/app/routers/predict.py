"""
Predict Router — Ana tahmin pipeline'ı.
POST /predict    → Tek e-posta analizi
POST /predict/batch → Toplu analiz
"""
from fastapi import APIRouter, BackgroundTasks
from app.schemas.request import EmailRequest, BatchEmailRequest
from app.schemas.response import PredictionResponse
from app.agents.agent1_ingestion import ingest_email
from app.agents.agent2_preprocessing import parallel_preprocess
from app.agents.agent3_stylometric import extract_stylometric_features, features_to_vector
from app.agents.agent4_xgboost import xgboost_predict
from app.agents.agent5_threshold import adaptive_threshold_decision
from app.agents.agent6_nlp import nlp_preprocess
from app.agents.agent7_dual_branch import dual_branch_predict
from app.agents.agent8_meta_classifier import meta_classify
from app.agents.agent9_output import format_output
import time
import uuid
import asyncio

router = APIRouter()


async def _run_pipeline(request: EmailRequest) -> dict:
    """Ana 9-agent pipeline orkestrasyon."""
    start = time.time()
    email_id = request.email_id or str(uuid.uuid4())

    # ── Agent 1: Email Ingestion ──
    email_data = await ingest_email(
        subject=request.subject,
        body=request.body,
        sender=request.sender,
        email_id=email_id
    )

    # ── Agent 2: Paralel Preprocessing (Thread A: Clean + Thread B: SHA-256) ──
    preprocess_result = await parallel_preprocess(
        email_data["body_plain"],
        email_data["headers"]["subject"]
    )

    # ── Agent 3: Stylometric Feature Extraction (HAM metinden!) ──
    stylo_features = extract_stylometric_features(email_data["body_plain"])
    feature_vector = features_to_vector(stylo_features)

    # ── Agent 4: XGBoost Stacking (Level-0) ──
    xgb_result = await xgboost_predict(feature_vector)

    # ── Agent 5: Adaptive Threshold Decision ──
    decision = adaptive_threshold_decision(xgb_result["probability"])

    detection_path = "FAST_PATH"
    lstm_score = None
    final_prob = xgb_result["probability"]

    if decision["decision"] == "CERTAIN/PHISHING":
        verdict = "PHISHING"
    elif decision["decision"] == "CERTAIN/SAFE":
        verdict = "SAFE"
    else:
        # ── UNCERTAIN → Derin Analiz Pipeline ──
        detection_path = "FULL_PIPELINE"

        # ── Agent 6: NLP Preprocessing ──
        nlp_result = await nlp_preprocess(preprocess_result["cleaned_text"])

        # ── Agent 7: Dual Branch (GloVe + BiLSTM + Attention) ──
        lstm_result = await dual_branch_predict(nlp_result["padded_sequence"])
        lstm_score = lstm_result["probability"]

        # ── Agent 8: Meta-Classifier (Stacking Level-2) ──
        meta_result = await meta_classify(xgb_result["probability"], lstm_score)
        final_prob = meta_result["probability"]
        verdict = "PHISHING" if meta_result["prediction"] == 1 else "SAFE"

    latency = (time.time() - start) * 1000

    # ── Agent 9: Output & Logging ──
    output = await format_output(
        email_id=email_id,
        verdict=verdict,
        probability=final_prob,
        xgb_score=xgb_result["probability"],
        lstm_score=lstm_score,
        detection_path=detection_path,
        processing_time_ms=latency,
        stylo_features=stylo_features,
    )

    return output


@router.post("/predict", response_model=PredictionResponse)
async def predict_email(request: EmailRequest):
    """Tek e-posta phishing analizi."""
    result = await _run_pipeline(request)
    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(request: BatchEmailRequest):
    """Toplu e-posta analizi (max 100)."""
    tasks = [_run_pipeline(email) for email in request.emails]
    results = await asyncio.gather(*tasks)
    return [PredictionResponse(**r) for r in results]
