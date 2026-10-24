from __future__ import annotations

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from src.config import CNNModelConfig, ExperimentConfig, ModelConfig, OptimConfig
from src.data_module import MNISTDataModule
from src.model import _ACTIVATIONS, _OPTIMIZERS, CNNClassifier, MLPClassifier, build_model, load_model

BATCH = 4
REPO_ROOT = Path(__file__).resolve().parent.parent


def _images(batch: int = BATCH) -> torch.Tensor:
    return torch.rand(batch, 1, 28, 28)


def _config(architecture: str, **optim) -> ExperimentConfig:
    model_config = CNNModelConfig() if architecture == "cnn" else ModelConfig()
    return ExperimentConfig(architecture=architecture, model=model_config, optim=OptimConfig(**optim))


# --- Forward pass -----------------------------------------------------------


@pytest.mark.parametrize("hidden_dims", [[64], [256, 128, 64]])
@pytest.mark.parametrize("activation", list(_ACTIVATIONS))
def test_mlp_output_shape(hidden_dims, activation):
    model = MLPClassifier(ModelConfig(hidden_dims=hidden_dims, activation=activation), OptimConfig())
    assert model(_images()).shape == (BATCH, 10)


@pytest.mark.parametrize("conv_channels", [[16], [32, 64], [16, 32, 64]])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_cnn_output_shape(conv_channels, kernel_size):
    # 3 conv blocks shrink 28 -> 14 -> 7 -> 3, which checks the flat_dim arithmetic for an odd spatial size.
    model = CNNClassifier(CNNModelConfig(conv_channels=conv_channels, kernel_size=kernel_size), OptimConfig())
    assert model(_images()).shape == (BATCH, 10)


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_eval_mode_is_deterministic(architecture):
    """Dropout must switch off in eval mode, or predictions would be random."""
    model = build_model(_config(architecture)).eval()
    x = _images()
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x))


# --- Dispatch and optimizer --------------------------------------------------


def test_build_model_dispatches_on_architecture():
    assert type(build_model(_config("mlp"))) is MLPClassifier
    assert type(build_model(_config("cnn"))) is CNNClassifier


@pytest.mark.parametrize("optimizer", list(_OPTIMIZERS))
def test_configure_optimizers_uses_configured_optimizer(optimizer):
    model = build_model(_config("mlp", optimizer=optimizer, lr=0.01, weight_decay=0.001))
    opt = model.configure_optimizers()
    assert type(opt) is _OPTIMIZERS[optimizer]
    assert opt.defaults["lr"] == 0.01
    assert opt.defaults["weight_decay"] == 0.001


def test_label_smoothing_reaches_loss():
    model = build_model(_config("mlp", label_smoothing=0.1))
    assert model.loss_fn.label_smoothing == 0.1


# --- Checkpoints ------------------------------------------------------------


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_checkpoint_round_trip(architecture, data_config, tmp_path):
    config = _config(architecture)
    config.data = data_config
    model = build_model(config)
    trainer = pl.Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_progress_bar=False, default_root_dir=tmp_path
    )
    trainer.fit(model, datamodule=MNISTDataModule(data_config))
    ckpt_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt_path)

    # The PyTorch 2.6 fix: hyperparameters are stored as plain dicts, so the
    # checkpoint must open under the strict default, with no custom classes.
    torch.load(ckpt_path, weights_only=True)

    reloaded = load_model(config, str(ckpt_path))
    assert type(reloaded) is type(model)
    assert reloaded.model_config == model.model_config
    x = _images()
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), model(x))


@pytest.mark.parametrize(
    ("config_path", "ckpt_path"),
    [
        ("configs/best_from_search.yaml", "results/mlp_best_from_search.ckpt"),
        ("configs/cnn_best_from_search.yaml", "results/cnn_best_from_search.ckpt"),
    ],
)
def test_committed_checkpoint_matches_its_config(config_path, ckpt_path):
    """The README's results come from these files; they must stay loadable and in sync with their YAML."""
    ckpt = REPO_ROOT / ckpt_path
    if not ckpt.exists():
        pytest.skip(f"{ckpt_path} not present")
    config = ExperimentConfig.from_yaml(REPO_ROOT / config_path)
    model = load_model(config, str(ckpt))
    assert model.model_config == config.model
    assert model.optim_config == config.optim
