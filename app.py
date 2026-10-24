"""Web demo: draw a digit, see the tuned CNN's top-3 prediction.

Run with `python app.py` (after `pip install -r requirements-demo.txt`) and open
the printed local address. It reuses src/predict.py, so the demo applies exactly
the preprocessing the CLI does and the tests cover.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import gradio as gr

from src.config import ExperimentConfig
from src.model import BaseClassifier, load_model
from src.predict import predict

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "cnn_best_from_search.yaml"
CHECKPOINT = ROOT / "results" / "cnn_best_from_search.ckpt"
GITHUB_URL = "https://github.com/ivannamor/mnist-digit-classification"


@cache
def get_model() -> BaseClassifier:
    return load_model(ExperimentConfig.from_yaml(CONFIG), str(CHECKPOINT))


def classify(drawing: dict | None) -> dict[str, float] | None:
    """Sketchpad value -> {digit: probability}, or None while the canvas is empty."""
    image = drawing.get("composite") if drawing else None
    if image is None:
        return None
    try:
        probabilities = predict(get_model(), image)
    except ValueError:  # blank canvas, e.g. right after clearing it
        return None
    return {str(digit): float(p) for digit, p in enumerate(probabilities)}


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MNIST digit classifier") as demo:
        gr.Markdown(
            "# Handwritten digit classifier\n"
            "Draw a single digit (0-9). A CNN trained on MNIST (99.2% test accuracy) predicts it live. "
            f"Code, training and evaluation: [GitHub]({GITHUB_URL})."
        )
        with gr.Row():
            canvas = gr.Sketchpad(
                type="pil",
                canvas_size=(280, 280),
                brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=18),
                label="Draw here",
            )
            result = gr.Label(num_top_classes=3, label="Prediction")
        canvas.change(classify, inputs=canvas, outputs=result, trigger_mode="always_last")
        gr.Markdown(
            "Before predicting, the drawing is converted the way MNIST was built: inverted to light-on-dark, "
            "cropped, scaled to fit 20x20 and centred by centre of mass in 28x28. Probabilities top out "
            "around 90-95% because the model was trained with label smoothing."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
