"""Regression tests: the committed checkpoints, evaluated on the real Kaggle
data, must still reproduce the numbers published in the README (the results
table and the seeded distribution-shift table), and predict() must still
recognise real digits redrawn as photo-like images.

Needs data/train.csv, so it is skipped in CI and on fresh clones; run it
locally after any change to data loading, cleaning, splitting or the model.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import pytorch_lightning as pl
from PIL import Image

from src.config import ExperimentConfig
from src.data_module import MNISTDataModule
from src.model import load_model
from src.predict import predict
from src.robustness import run_noise_location_check, run_shift_robustness, run_translation_curve
from tests.conftest import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"

pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(not (DATA_DIR / "train.csv").exists(), reason="needs the real Kaggle data in data/train.csv"),
]

# Exactly as published in the README results table (percent, 2 decimals).
README_RESULTS = {
    "mlp": {
        "config": "configs/best_from_search.yaml",
        "checkpoint": "results/mlp_best_from_search.ckpt",
        "test_accuracy": 98.30,
        "test_precision": 98.29,
        "test_recall": 98.30,
    },
    "cnn": {
        "config": "configs/cnn_best_from_search.yaml",
        "checkpoint": "results/cnn_best_from_search.ckpt",
        "test_accuracy": 99.21,
        "test_precision": 99.22,
        "test_recall": 99.21,
    },
    # The same configurations retrained with augmentation (README augmentation section).
    "mlp_augmented": {
        "config": "configs/augmented.yaml",
        "checkpoint": "results/augmented.ckpt",
        "test_accuracy": 98.21,
        "test_precision": 98.19,
        "test_recall": 98.21,
    },
    "cnn_augmented": {
        "config": "configs/cnn_augmented.yaml",
        "checkpoint": "results/cnn_augmented.ckpt",
        "test_accuracy": 99.25,
        "test_precision": 99.25,
        "test_recall": 99.25,
    },
}


def _model_and_test_loader(architecture: str):
    expected = README_RESULTS[architecture]
    config = ExperimentConfig.from_yaml(REPO_ROOT / expected["config"])
    data_module = MNISTDataModule(replace(config.data, data_dir=str(DATA_DIR), num_workers=0))
    data_module.setup()
    return load_model(config, str(REPO_ROOT / expected["checkpoint"])), data_module.test_dataloader()


@pytest.mark.parametrize("architecture", list(README_RESULTS))
def test_committed_checkpoint_reproduces_readme_results(architecture):
    expected = README_RESULTS[architecture]
    config = ExperimentConfig.from_yaml(REPO_ROOT / expected["config"])
    # Absolute data path so the test works from any working directory.
    data_config = replace(config.data, data_dir=str(DATA_DIR), num_workers=0)

    model = load_model(config, str(REPO_ROOT / expected["checkpoint"]))
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
    (metrics,) = trainer.test(model, datamodule=MNISTDataModule(data_config), verbose=False)

    # Compare at the README's own precision: a mismatch means either the
    # pipeline changed behaviour or the README is out of date.
    for name in ("test_accuracy", "test_precision", "test_recall"):
        assert round(metrics[name] * 100, 2) == expected[name], name


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])  # the models predict.py and the demo use
def test_predict_handles_photo_like_images(architecture):
    """Real test digits redrawn like a user's image (dark ink on white, 4x
    larger, at a random spot on a 200x200 page) must still be recognised
    through predict()'s preprocessing. Without it, accuracy on these images
    is ~29% (CNN) / ~15% (MLP): the translation weakness from the robustness checks.
    """
    expected = README_RESULTS[architecture]
    config = ExperimentConfig.from_yaml(REPO_ROOT / expected["config"])
    data_module = MNISTDataModule(replace(config.data, data_dir=str(DATA_DIR), num_workers=0))
    data_module.setup()
    model = load_model(config, str(REPO_ROOT / expected["checkpoint"]))

    rng = np.random.default_rng(0)
    n = 1000
    correct = 0
    for pixels, label in zip(data_module.test_dataset.pixels[:n], data_module.test_dataset.labels[:n], strict=True):
        digit = Image.fromarray((255 - pixels.reshape(28, 28)).astype(np.uint8)).resize((112, 112))
        page = Image.new("L", (200, 200), "white")
        page.paste(digit, tuple(int(v) for v in rng.integers(0, 200 - 112, size=2)))
        correct += predict(model, page).argmax().item() == label

    assert correct / n >= 0.98


# Exactly as published in the README's distribution-shift table (plain accuracy, %).
README_SHIFT_RESULTS = {
    "mlp": {"clean": 98.31, "rotate_15deg": 95.46, "rotate_30deg": 80.76, "translate_4px": 9.42,
            "noise_sigma0.1": 90.33, "noise_sigma0.3": 41.68},
    "cnn": {"clean": 99.23, "rotate_15deg": 98.10, "rotate_30deg": 90.14, "translate_4px": 21.24,
            "noise_sigma0.1": 99.23, "noise_sigma0.3": 78.50},
    "mlp_augmented": {"clean": 98.21, "rotate_15deg": 96.92, "rotate_30deg": 88.57, "translate_4px": 94.89,
                      "noise_sigma0.1": 70.90, "noise_sigma0.3": 15.25},
    "cnn_augmented": {"clean": 99.25, "rotate_15deg": 98.39, "rotate_30deg": 93.54, "translate_4px": 98.29,
                      "noise_sigma0.1": 91.05, "noise_sigma0.3": 13.39},
}


@pytest.mark.parametrize("architecture", list(README_SHIFT_RESULTS))
def test_shift_robustness_reproduces_readme_table(architecture):
    model, loader = _model_and_test_loader(architecture)
    results = run_shift_robustness(model, loader)
    assert {name: round(value * 100, 2) for name, value in results.items()} == README_SHIFT_RESULTS[architecture]


# README augmentation section: translation curve, shift 0..8 px (plain accuracy, %, 1 decimal).
README_TRANSLATION_CURVE = {
    "mlp": [98.3, 91.2, 58.8, 25.8, 9.4, 5.9, 6.1, 7.4, 8.3],
    "mlp_augmented": [98.2, 98.1, 98.1, 97.9, 94.9, 76.5, 35.4, 14.4, 9.9],
    "cnn": [99.2, 98.4, 87.1, 48.3, 21.2, 13.8, 13.7, 13.9, 14.0],
    "cnn_augmented": [99.2, 99.2, 99.2, 99.1, 98.3, 92.1, 62.4, 24.5, 12.8],
}


@pytest.mark.parametrize("architecture", list(README_TRANSLATION_CURVE))
def test_translation_curve_reproduces_readme(architecture):
    model, loader = _model_and_test_loader(architecture)
    curve = run_translation_curve(model, loader)
    assert [round(value * 100, 1) for value in curve.values()] == README_TRANSLATION_CURVE[architecture]


# README augmentation section: noise sigma=0.3 by location (plain accuracy, %, 1 decimal).
README_NOISE_LOCATION = {
    "mlp": {"everywhere": 41.7, "border_only": 82.2, "center_only": 60.1},
    "mlp_augmented": {"everywhere": 15.2, "border_only": 44.2, "center_only": 49.2},
    "cnn": {"everywhere": 78.5, "border_only": 99.0, "center_only": 82.4},
    "cnn_augmented": {"everywhere": 13.4, "border_only": 86.4, "center_only": 31.4},
}


@pytest.mark.parametrize("architecture", list(README_NOISE_LOCATION))
def test_noise_location_check_reproduces_readme(architecture):
    model, loader = _model_and_test_loader(architecture)
    results = run_noise_location_check(model, loader)
    assert {name: round(value * 100, 1) for name, value in results.items()} == README_NOISE_LOCATION[architecture]
