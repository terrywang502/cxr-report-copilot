#!/bin/bash
# Start FastAPI in background on port 8000
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to be ready
sleep 5

# Start Gradio on port 7860 (HF Spaces public port)
python gradio_app.py
