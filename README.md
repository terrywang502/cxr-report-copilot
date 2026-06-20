# CXR Report Copilot

**A deployed, multimodal chest X-ray triage assistant** — classifies thoracic pathologies, and is being extended with explainability (Grad-CAM) and LLM-generated draft reports (Claude API).

🔗 **Live Demo:** [terrywang502-cxr-report-copilot.hf.space/docs](https://terrywang502-cxr-report-copilot.hf.space/docs)

> ⚠️ For research/demo purposes only — not for clinical use.

---

## Status

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Model inference API (FastAPI) + Docker containerization + public deployment | ✅ **Complete** |
| **Phase 2** | Grad-CAM explainability — visualize model attention over X-ray findings | ✅ **Complete** |
| **Phase 3** | LLM-generated draft radiology reports via Claude API | 🔜 Planned |
| **Phase 4** | Interactive front-end (upload → prediction → heatmap → report) | 🔜 Planned |

This project is being built incrementally and shipped at each milestone rather than held back until "finished" — Phase 1 is live and usable today.

---

## What it does

Upload a chest X-ray and the API returns classification probabilities across four categories: **COVID-19, Lung Opacity, Normal, Viral Pneumonia.**

```bash
curl -X POST 'https://terrywang502-cxr-report-copilot.hf.space/predict' \
     -F 'file=@xray.png'
```

```json
{
  "prediction": "Normal",
  "confidence": 0.92,
  "probabilities": {
    "COVID": 0.03,
    "Lung_Opacity": 0.02,
    "Normal": 0.92,
    "Viral Pneumonia": 0.03
  },
  "disclaimer": "For research/demo purposes only — not for clinical use."
}
```

---

## Model

A three-stage transfer learning pipeline built on **EfficientNet-B0**:

- **Stage 1** — Feature extraction with a frozen EfficientNet-B0 backbone
- **Stage 2** — Fine-tuning with the last two layers unfrozen
- **Stage 3** — Multi-task learning: added a pixel segmentation head (lung masks) alongside classification, with a combined loss function

**Results (Stage 3, held-out test set):**

| Metric | Score |
|---|---|
| Accuracy | 95% |
| Macro F1 | 0.96 |

Trained on the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (21,165 images, 4 classes), with stratified sampling, data augmentation, and weighted cross-entropy loss to handle class imbalance.

---

## Architecture

```
X-ray upload
     │
     ▼
EfficientNet-B0 (multi-task: classification + segmentation)
     │
     ▼
Classification probabilities (4 classes)
     │
     ▼
FastAPI inference endpoint
     │
     ▼
Docker container → deployed on Hugging Face Spaces
```

---

## Tech stack

`Python` · `PyTorch` · `EfficientNet-B0` · `Albumentations` · `FastAPI` · `Docker` · `Hugging Face Spaces`

---

## Run locally

```bash
# Clone and install
git clone https://github.com/terrywang502/cxr-report-copilot.git
cd cxr-report-copilot
pip install -r requirements.txt

# Run the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Test it at http://localhost:8000/docs
```

> Pretrained weights are not included in this repo (kept out via `.gitignore` due to file size). The live demo above runs the full model — to retrain from scratch, see `predict.py` for the model architecture and training notes.

---

## Project structure

```
cxr-report-copilot/
├── predict.py          # Model definition + inference logic
├── app.py               # FastAPI application
├── Dockerfile            # Container definition
├── requirements.txt
└── README.md
```

---

## Why this project

Most medical ML portfolio projects stop at a notebook with an accuracy score. This one goes further: a trained model wrapped in a production API, containerized, and deployed to a public endpoint — with explainability and LLM-assisted reporting layered on next. The goal is to demonstrate not just modeling ability, but the engineering required to make a model usable.

---

## Author

**Huaizeng (Terry) Wang** — Master of Data Science, Memorial University of Newfoundland
[LinkedIn](https://linkedin.com/in/terry-wang-767a53382) · [GitHub](https://github.com/terrywang502)
