"""Regression test: the committed checkpoints, evaluated on the real Kaggle
data, must still reproduce the test-set numbers published in the README.

Needs data/train.csv, so it is skipped in CI and on fresh clones; run it
locally after any change to data loading, cleaning, splitting or the model.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import pytorch_lightning as pl

from src.config import ExperimentConfig
from src.data_module import MNISTDataModule
from src.model import load_model
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
}


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
