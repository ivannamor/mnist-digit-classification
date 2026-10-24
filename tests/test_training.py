"""End-to-end smoke tests: every entry point actually runs, on tiny synthetic
data, for 1 epoch, on CPU. They check wiring, not accuracy.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import optuna
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import CNNModelConfig, DataConfig, ExperimentConfig, ModelConfig, OptimConfig, TrainerConfig
from src.data_module import MNISTDataModule
from src.evaluate import evaluate
from src.model import build_model
from src.robustness import _PERTURBATIONS, run_cross_seed_trials, run_shift_robustness
from src.search import objective
from src.train import train
from tests.test_configs import _valid_params

matplotlib.use("Agg")  # render plots to files only; no GUI window in tests/CI

ARCHITECTURES = ["mlp", "cnn"]


def _tiny_config(architecture: str, data_config: DataConfig, **optim) -> ExperimentConfig:
    if architecture == "cnn":
        model_config = CNNModelConfig(conv_channels=[8, 16], fc_hidden_dim=32)
    else:
        model_config = ModelConfig(hidden_dims=[32])
    return ExperimentConfig(
        architecture=architecture,
        data=data_config,
        model=model_config,
        optim=OptimConfig(**optim),
        trainer=TrainerConfig(max_epochs=1, accelerator="cpu", patience=1, experiment_name="test"),
    )


def _trainer(**kwargs) -> pl.Trainer:
    return pl.Trainer(
        accelerator="cpu", logger=False, enable_progress_bar=False, enable_checkpointing=False, **kwargs
    )


# --- Training loop wiring ---------------------------------------------------


@pytest.mark.parametrize("lr_scheduler", [None, "onecycle", "plateau"])
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_fit_runs_with_every_scheduler(architecture, lr_scheduler, data_config):
    config = _tiny_config(architecture, data_config, lr_scheduler=lr_scheduler)
    model = build_model(config)
    trainer = _trainer(max_epochs=2)
    trainer.fit(model, datamodule=MNISTDataModule(data_config))

    for name in ("train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall"):
        assert name in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    assert 0.0 <= trainer.callback_metrics["val_accuracy"] <= 1.0


def test_onecycle_actually_changes_the_learning_rate(data_config):
    config = _tiny_config("mlp", data_config, lr_scheduler="onecycle", lr=0.01)
    model = build_model(config)
    trainer = _trainer(max_epochs=1)
    trainer.fit(model, datamodule=MNISTDataModule(data_config))
    # OneCycle ends far below max_lr; a scheduler that was never stepped would leave lr at 0.01.
    assert trainer.optimizers[0].param_groups[0]["lr"] < 0.01 / 10


@pytest.mark.filterwarnings("ignore:You defined a `validation_step` but have no `val_dataloader`")
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_training_reduces_loss(architecture, data_config):
    """The most basic learning check: repeated steps on one fixed batch must
    overfit it. Seeded, since otherwise the init depends on which tests ran before.
    """
    pl.seed_everything(0)
    config = _tiny_config(architecture, data_config, lr=1e-2)
    model = build_model(config)
    data_module = MNISTDataModule(data_config)
    data_module.setup()
    x, y = next(iter(data_module.train_dataloader()))

    model.eval()
    with torch.no_grad():
        before = model.loss_fn(model(x), y).item()
    model.train()  # Lightning keeps the module's current mode, so undo the eval() above before fitting
    one_batch = DataLoader(TensorDataset(x, y), batch_size=len(y))
    _trainer(max_epochs=30, limit_val_batches=0).fit(model, train_dataloaders=one_batch)
    model.eval()
    with torch.no_grad():
        after = model.loss_fn(model(x), y).item()
    assert after < before * 0.5


# --- Entry points (train.py, evaluate.py, search.py) -----------------------


def test_train_saves_best_checkpoint_and_never_touches_test_set(data_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # train() writes outputs/logs/ relative to the working directory
    trainer = train(_tiny_config("mlp", data_config))

    best = Path(trainer.checkpoint_callback.best_model_path).resolve()
    assert best.is_file()
    assert best.is_relative_to(tmp_path.resolve())
    assert not [name for name in trainer.callback_metrics if name.startswith("test_")]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_evaluate_writes_plots(architecture, data_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _tiny_config(architecture, data_config)
    checkpoint = train(config).checkpoint_callback.best_model_path

    output_dir = tmp_path / "evaluation"
    evaluate(config, checkpoint, output_dir)

    assert (output_dir / "confusion_matrix.png").stat().st_size > 0
    # An untrained-ish model on random pixels misclassifies plenty, so this plot must exist too.
    assert (output_dir / "misclassified_examples.png").stat().st_size > 0


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_search_objective_returns_validation_accuracy(architecture, data_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trial = optuna.trial.FixedTrial(_valid_params(architecture, batch_size=32))
    score = objective(trial, _tiny_config(architecture, data_config))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # Trials only report a score; a checkpoint per trial would pile up on disk.
    assert not list(tmp_path.rglob("*.ckpt"))


# --- Robustness experiments ------------------------------------------------


@pytest.mark.parametrize("name", list(_PERTURBATIONS))
def test_perturbations_keep_shape_and_range(name):
    torch.manual_seed(0)
    images = torch.rand(8, 1, 28, 28)
    perturbed = _PERTURBATIONS[name](images)
    assert perturbed.shape == images.shape
    assert perturbed.min() >= 0.0
    assert perturbed.max() <= 1.0
    assert not torch.equal(perturbed, images), "perturbation had no effect"


def test_translate_moves_pixels_by_4():
    image = torch.zeros(1, 1, 28, 28)
    image[0, 0, 10, 10] = 1.0
    shifted = _PERTURBATIONS["translate_4px"](image)
    assert shifted[0, 0, 14, 14] == 1.0
    assert shifted.sum() == 1.0


def test_shift_robustness_reports_every_perturbation(data_config):
    data_module = MNISTDataModule(data_config)
    data_module.setup()
    model = build_model(_tiny_config("cnn", data_config))
    results = run_shift_robustness(model, data_module.test_dataloader())

    assert set(results) == {"clean", *_PERTURBATIONS}
    assert all(0.0 <= accuracy <= 1.0 for accuracy in results.values())


def test_cross_seed_trials_one_result_per_seed(data_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = run_cross_seed_trials(_tiny_config("mlp", data_config), training_seeds=[1, 2])
    assert [r["training_seed"] for r in results] == [1, 2]
    for r in results:
        assert 0.0 <= r["val_accuracy"] <= 1.0
        assert 0.0 <= r["test_accuracy"] <= 1.0
    assert not list(tmp_path.rglob("*.ckpt")), "docstring promises no checkpoints are saved"
