# ── CXR Report Copilot — Dockerfile ──────────────────────────────────────────
# Targets Hugging Face Spaces (port 7860) and any standard Docker host.
# CPU-only by default; swap the base image for GPU if needed.

FROM python:3.10-slim

# System deps (OpenCV / Albumentations may need libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY predict.py .
COPY app.py .

# Place your trained weights here at build time OR mount at runtime.
# To bake the model into the image (simplest for HF Spaces):
#   COPY stage3_multitask_best.pt .
# To mount at runtime:
#   docker run -v /path/to/stage3_multitask_best.pt:/app/stage3_multitask_best.pt ...
COPY stage3_multitask_best.pt .

# HF Spaces expects port 7860; change to 8000 for local dev if you prefer
ENV MODEL_PATH=/app/stage3_multitask_best.pt
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
