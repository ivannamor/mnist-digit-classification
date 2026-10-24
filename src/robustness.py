"""Robustness experiments beyond the core train/search/evaluate pipeline:

- Cross-seed trials: does the reported accuracy hold up across different
  random initializations, or was it a lucky run?
- Distribution-shift evaluation: does an already-trained model hold up on
  perturbed versions of the test set (rotation, translation, noise), or is
  it only good at MNIST's own narrow, curated distribution?
"""
from __future__ import annotations

import copy
from collections.abc import Callable

import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data_module import MNISTDataModule
from src.model import BaseClassifier, build_model


def run_cross_seed_trials(config: ExperimentConfig, training_seeds: list[int]) -> list[dict]:
    """Retrains `config` once per training seed, with `config.data.split_seed`
    held fixed -- so every trial trains/evaluates on the identical split, and
    only random init/shuffling varies. No checkpoints are saved (mirrors
    search.py's objective()) -- only the resulting metrics matter here.
    """
    results = []
    for training_seed in training_seeds:
        trial_config = copy.deepcopy(config)
        trial_config.trainer.training_seed = training_seed
        pl.seed_everything(training_seed)

        data_module = MNISTDataModule(trial_config.data)
        model = build_model(trial_config)
        trainer = pl.Trainer(
            max_epochs=trial_config.trainer.max_epochs,
            accelerator=trial_config.trainer.accelerator,
            devices=trial_config.trainer.devices,
            callbacks=[EarlyStopping(monitor="val_loss", patience=trial_config.trainer.patience)],
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, datamodule=data_module)
        val_accuracy = trainer.callback_metrics["val_accuracy"].item()

        trainer.test(model, datamodule=data_module, verbose=False)
        test_accuracy = trainer.callback_metrics["test_accuracy"].item()

        results.append(
            {"training_seed": training_seed, "val_accuracy": val_accuracy, "test_accuracy": test_accuracy}
        )
    return results


# Each perturbation is applied to a whole batch of (B, 1, 28, 28) tensors
# already normalized to [0, 1]. Two severities for rotation/noise, so we can
# see whether accuracy degrades gracefully or falls off a cliff.
_PERTURBATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "rotate_15deg": lambda x: TF.rotate(x, angle=15),
    "rotate_30deg": lambda x: TF.rotate(x, angle=30),
    "translate_4px": lambda x: TF.affine(x, angle=0.0, translate=[4, 4], scale=1.0, shear=0.0),
    "noise_sigma0.1": lambda x: (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0),
    "noise_sigma0.3": lambda x: (x + torch.randn_like(x) * 0.3).clamp(0.0, 1.0),
}


def _accuracy_under(
    model: BaseClassifier, dataloader: DataLoader, perturbation: Callable[[torch.Tensor], torch.Tensor]
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            preds = model(perturbation(images)).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run_shift_robustness(model: BaseClassifier, dataloader: DataLoader) -> dict[str, float]:
    """Evaluates an already-trained model on the clean test set and on each
    perturbed version -- no retraining, just measuring how far accuracy
    drops once the input no longer looks like MNIST's own curated data.
    """
    results = {"clean": _accuracy_under(model, dataloader, lambda x: x)}
    for name, perturbation in _PERTURBATIONS.items():
        results[name] = _accuracy_under(model, dataloader, perturbation)
    return results
