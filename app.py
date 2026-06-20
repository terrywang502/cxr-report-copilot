"""
app.py — FastAPI serving layer for CXR Report Copilot

Endpoints:
    GET  /health     → liveness check
    POST /predict    → upload a chest X-ray, get classification probabilities

⚠️  DISCLAIMER: For research / demo purposes only — NOT for clinical use.
"""

import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from predict import IDX2LABEL, MultiTaskModel, load_model, predict

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CXR Report Copilot",
    description=(
        "Chest X-ray triage assistant — classifies thoracic pathologies "
        "(COVID-19, Lung Opacity, Normal, Viral Pneumonia) using a multi-task "
        "EfficientNetB0 trained with joint classification + lung segmentation.\n\n"
        "⚠️ For research/demo purposes only — not for clinical use."
    ),
    version="1.0.0",
)

# ── Globals ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "stage3_multitask_best.pt")
_model: MultiTaskModel | None = None


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global _model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}\n"
            "Set MODEL_PATH env var or place stage3_multitask_best.pt in the working directory."
        )
    _model = load_model(MODEL_PATH, DEVICE)
    print(f"✅ Model loaded from '{MODEL_PATH}' on {DEVICE}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "device": str(DEVICE), "model_loaded": _model is not None}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image (PNG / JPEG) and receive classification results.

    Returns JSON with:
    - prediction: top predicted class
    - confidence: probability of top class (0–1)
    - probabilities: softmax scores for all 4 classes
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

    return JSONResponse(
        content={
            **result,
            "disclaimer": "For research/demo purposes only — not for clinical use.",
        }
    )
