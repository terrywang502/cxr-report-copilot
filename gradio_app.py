"""
gradio_app.py — Gradio frontend for CXR Report Copilot
"""

import base64
import io
import os

import gradio as gr
import httpx
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

LABEL_COLORS = {
    "COVID": "#e74c3c",
    "Lung_Opacity": "#e67e22",
    "Normal": "#27ae60",
    "Viral Pneumonia": "#8e44ad",
}

DISCLAIMER = (
    "⚠️ For research/demo purposes only — NOT for clinical use. "
    "Always consult a qualified radiologist."
)


def predict(image: Image.Image):
    if image is None:
        return None, None, "Please upload a chest X-ray image.", DISCLAIMER

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    try:
        response = httpx.post(
            API_URL,
            files={"file": ("xray.png", buf, "image/png")},
            data={"include_heatmap": "true", "include_report": "true"},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.ConnectError:
        return None, "❌ Cannot connect to API.", "Is uvicorn running?", DISCLAIMER
    except Exception as e:
        return None, f"❌ Error: {str(e)}", "", DISCLAIMER

    prediction = data.get("prediction", "Unknown")
    confidence = data.get("confidence", 0)
    probabilities = data.get("probabilities", {})
    draft_report = data.get("draft_report", "Report unavailable.")
    heatmap_b64 = data.get("heatmap_base64", "")

    heatmap_img = None
    if heatmap_b64:
        heatmap_bytes = base64.b64decode(heatmap_b64)
        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))

    color = LABEL_COLORS.get(prediction, "#2c3e50")
    summary_html = f"""
    <div style="font-family: Arial, sans-serif; padding: 16px;
                border-radius: 8px; background: #f8f9fa;">
        <h2 style="color: {color}; margin: 0 0 8px 0;">{prediction}</h2>
        <p style="font-size: 18px; margin: 0 0 16px 0;">
            Confidence: <strong>{confidence:.1%}</strong>
        </p>
        <hr style="border: none; border-top: 1px solid #dee2e6; margin: 12px 0;">
        <p style="font-size: 13px; color: #6c757d; margin: 0 0 8px 0;">
            All class probabilities:
        </p>
    """
    for label, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        bar_color = LABEL_COLORS.get(label, "#6c757d")
        bar_width = int(prob * 100)
        summary_html += f"""
        <div style="margin: 6px 0;">
            <div style="display: flex; justify-content: space-between;
                        font-size: 13px; margin-bottom: 3px;">
                <span>{label}</span><span>{prob:.1%}</span>
            </div>
            <div style="background: #e9ecef; border-radius: 4px; height: 8px;">
                <div style="background: {bar_color}; width: {bar_width}%;
                            height: 8px; border-radius: 4px;"></div>
            </div>
        </div>
        """
    summary_html += "</div>"

    report_text = draft_report.strip() if draft_report else "Report unavailable."
    return heatmap_img, summary_html, report_text, DISCLAIMER


with gr.Blocks(title="CXR Report Copilot") as demo:
    gr.Markdown("# 🫁 CXR Report Copilot")
    gr.Markdown("AI-powered chest X-ray triage — classification · Grad-CAM · LLM report")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Chest X-ray (PNG / JPEG)", height=300)
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")
            gr.Markdown("> ⚠️ For research/demo purposes only — NOT for clinical use.")
        with gr.Column(scale=1):
            heatmap_output = gr.Image(label="Grad-CAM Heatmap (model attention)", height=300)
            prediction_output = gr.HTML(label="Prediction")

    with gr.Row():
        report_output = gr.Textbox(label="Draft Radiology Report (AI-generated)", lines=10)

    disclaimer_output = gr.Textbox(visible=False)

    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[heatmap_output, prediction_output, report_output, disclaimer_output],
    )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
