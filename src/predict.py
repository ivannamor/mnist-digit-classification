"""Predict the digit in an arbitrary image (a photo, a scan, a drawing).

Real images don't look like MNIST, and the robustness checks show how little
it takes to break the models (a 4px shift drops the CNN to 21%). So images are
first converted the way MNIST itself was built: a light digit on a black
background, scaled to fit a 20x20 box, and placed in a 28x28 canvas with its
centre of mass at the centre.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.config import ExperimentConfig
from src.model import BaseClassifier, load_model

MNIST_SIZE = 28
DIGIT_BOX = 20
# Pixels darker than this fraction of the brightest one are treated as
# background when locating the digit, so faint noise doesn't widen the crop.
INK_THRESHOLD = 0.2


def preprocess(image: Image.Image) -> torch.Tensor:
    """Convert any image to a (1, 28, 28) float tensor in [0, 1], MNIST-style."""
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        # Transparent areas (e.g. a drawing canvas) become white paper.
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    pixels = np.asarray(ImageOps.grayscale(image), dtype=np.float32)

    # MNIST is light ink on black; most photos and drawings are dark on light.
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    if border.mean() > 127:
        pixels = 255.0 - pixels

    ink = pixels > INK_THRESHOLD * pixels.max()
    if pixels.max() == 0 or not ink.any():
        raise ValueError("no digit found in the image")
    rows, cols = np.nonzero(ink)
    digit = pixels[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    height, width = digit.shape
    scale = DIGIT_BOX / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(digit.astype(np.uint8)).resize(new_size, Image.Resampling.LANCZOS)
    digit = np.asarray(resized, dtype=np.float32)

    # Place the digit so its centre of mass lands on the canvas centre,
    # clamped so it always stays fully inside the canvas.
    height, width = digit.shape
    ys, xs = np.indices(digit.shape)
    total = digit.sum()
    center_y, center_x = (ys * digit).sum() / total, (xs * digit).sum() / total
    top = int(np.clip(round(MNIST_SIZE / 2 - center_y), 0, MNIST_SIZE - height))
    left = int(np.clip(round(MNIST_SIZE / 2 - center_x), 0, MNIST_SIZE - width))

    canvas = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.float32)
    canvas[top : top + height, left : left + width] = digit
    # Same scaling as training (MNISTCsvDataset divides raw pixels by 255).
    return torch.from_numpy(canvas / 255.0).view(1, MNIST_SIZE, MNIST_SIZE)


def predict(model: BaseClassifier, image: Image.Image) -> torch.Tensor:
    """Class probabilities, shape (10,), for one image."""
    model.eval()
    with torch.no_grad():
        logits = model(preprocess(image).unsqueeze(0))
    return logits.softmax(dim=-1).squeeze(0)


def _format(probabilities: torch.Tensor, top_k: int = 3) -> str:
    values, digits = probabilities.topk(top_k)
    ranking = ", ".join(f"{d}: {p:.1%}" for p, d in zip(values.tolist(), digits.tolist(), strict=True))
    return f"{digits[0].item()}   ({ranking})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the digit in one or more image files.")
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=Path("configs/cnn_best_from_search.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/cnn_best_from_search.ckpt"))
    args = parser.parse_args()

    model = load_model(ExperimentConfig.from_yaml(args.config), str(args.checkpoint))
    for path in args.images:
        with Image.open(path) as image:
            print(f"{path}: {_format(predict(model, image))}")


if __name__ == "__main__":
    main()
