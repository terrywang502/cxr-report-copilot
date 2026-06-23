"""
app.py — FastAPI serving layer for CXR Report Copilot

Endpoints:
    GET  /health     → liveness check
    POST /predict    → upload a chest X-ray, get classification +
                       Grad-CAM heatmap + LLM-generated draft report

⚠️  DISCLAIMER: For research / demo purposes only — NOT for clinical use.
"""

import io
import os

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
import json
from PIL import Image

from predict import IDX2LABEL, MultiTaskModel, load_model, predict, preprocess_image
from gradcam import generate_gradcam
from report_generator import generate_report

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CXR Report Copilot",
    description=(
        "Chest X-ray triage assistant — classifies thoracic pathologies "
        "(COVID-19, Lung Opacity, Normal, Viral Pneumonia) using a multi-task "
        "EfficientNetB0. Returns classification results, a Grad-CAM heatmap "
        "showing which regions drove the prediction, and an LLM-generated "
        "draft radiology report via Claude API.\n\n"
        "⚠️ For research/demo purposes only — not for clinical use."
    ),
    version="1.2.0",
)

# ── Globals ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "stage3_multitask_best.pt")
_model: MultiTaskModel | None = None
_label2idx: dict | None = None


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global _model, _label2idx
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}\n"
            "Set MODEL_PATH env var or place stage3_multitask_best.pt "
            "in the working directory."
        )
    _model = load_model(MODEL_PATH, DEVICE)
    _label2idx = {v: k for k, v in IDX2LABEL.items()}

    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    print(f"✅ Model loaded from '{MODEL_PATH}' on {DEVICE}")
    print(f"{'✅' if has_key else '⚠️ '} ANTHROPIC_API_KEY {'found' if has_key else 'NOT SET — report generation will be unavailable'}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness check."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": _model is not None,
        "report_generation_available": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    include_heatmap: bool = Form(True),
    include_report: bool = Form(True),
):
    """
    Upload a chest X-ray image (PNG / JPEG) and receive:
    - prediction: top predicted class
    - confidence: probability of top class (0–1)
    - probabilities: softmax scores for all 4 classes
    - heatmap_base64: Grad-CAM heatmap overlay (base64 PNG), if include_heatmap=True
    - draft_report: LLM-generated plain-language report, if include_report=True
    - disclaimer: mandatory clinical-use warning
    """
    # ── Validate content type ──────────────────────────────────────────────
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload PNG or JPEG.",
        )

    # ── Read & decode image ────────────────────────────────────────────────
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # ── Run inference ──────────────────────────────────────────────────────
    result = predict(image, _model, DEVICE)

    response = {
        **result,
        "disclaimer": "For research/demo purposes only — not for clinical use.",
    }

    # ── Grad-CAM heatmap ──────────────────────────────────────────────────
    if include_heatmap:
        target_idx = _label2idx[result["prediction"]]
        tensor = preprocess_image(image, DEVICE)
        response["heatmap_base64"] = generate_gradcam(
            image, tensor, _model, target_idx, DEVICE
        )

    # ── LLM draft report ──────────────────────────────────────────────────
    if include_report:
        if not os.getenv("ANTHROPIC_API_KEY"):
            response["draft_report"] = (
                "Report generation unavailable: ANTHROPIC_API_KEY is not configured."
            )
        else:
            try:
                response["draft_report"] = generate_report(
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    probabilities=result["probabilities"],
                )
            except Exception as exc:
                response["draft_report"] = f"Report generation failed: {str(exc)}"

    return Response(
    content=json.dumps(response, ensure_ascii=False),
    media_type="application/json"
)