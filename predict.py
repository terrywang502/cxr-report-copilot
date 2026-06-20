"""
predict.py — Single-image inference module for CXR Report Copilot
Extracted from capstone_project.ipynb (Stage 3 MultiTaskModel)

Usage:
    python predict.py path/to/xray.png
"""

import sys
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

# ── Label mapping ─────────────────────────────────────────────────────────────
# Matches label2idx = {label: i for i, label in enumerate(sorted(df['label'].unique()))}
# Sorted alphabetically: COVID(0), Lung Opacity(1), Normal(2), Viral Pneumonia(3)
IDX2LABEL = {
    0: "COVID",
    1: "Lung_Opacity",
    2: "Normal",
    3: "Viral Pneumonia",
}

# ── Model definition ──────────────────────────────────────────────────────────
# Copied verbatim from notebook Stage 3 MultiTaskModel, with pretrained=False → weights=None
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 4)

        self.backbone = base.features
        self.pool = base.avgpool
        self.dropout = base.classifier[0]
        self.classifier = nn.Linear(base.classifier[1].in_features, 4)

        self.seg_head = nn.Sequential(
            nn.Conv2d(1280, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        logits = self.classifier(self.dropout(pooled))
        seg = self.seg_head(feat)
        seg = nn.functional.interpolate(
            seg, size=(224, 224), mode="bilinear", align_corners=False
        )
        return logits, seg


# ── Transform ─────────────────────────────────────────────────────────────────
# Matches val_transform from notebook (resize + normalize only, no augmentation)
_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])


# ── Public API ────────────────────────────────────────────────────────────────
def load_model(model_path: str, device: torch.device) -> MultiTaskModel:
    """Load Stage 3 weights into MultiTaskModel and set to eval mode."""
    model = MultiTaskModel()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(image: Image.Image, model: MultiTaskModel, device: torch.device) -> dict:
    """
    Run inference on a single PIL image.

    Returns:
        {
            "prediction": str,           # top predicted class
            "confidence": float,         # probability of top class
            "probabilities": {           # softmax probabilities for all 4 classes
                "COVID": float,
                "Lung Opacity": float,
                "Normal": float,
                "Viral Pneumonia": float,
            }
        }
    """
    img_array = np.array(image.convert("RGB"))
    tensor = _transform(image=img_array)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    probabilities = {IDX2LABEL[i]: round(float(probs[i]), 4) for i in range(4)}
    top_label = max(probabilities, key=probabilities.get)

    return {
        "prediction": top_label,
        "confidence": probabilities[top_label],
        "probabilities": probabilities,
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    model_path = os.getenv("MODEL_PATH", "stage3_multitask_best.pt")
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if not image_path:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(model_path, device)
    image = Image.open(image_path)
    result = predict(image, model, device)

    print(f"\nPrediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.1%}")
    print("\nAll probabilities:")
    for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {label:<18} {prob:.1%}  {bar}")
