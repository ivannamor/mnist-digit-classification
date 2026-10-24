from __future__ import annotations

import warnings

import optuna
import pytest
import torch

from src.config import CNNModelConfig, ExperimentConfig, ModelConfig
from src.model import build_model
from src.search import (
    _CNN_CONV_CHANNELS_CHOICES,
    _MLP_HIDDEN_DIM_CHOICES,
    _sample_cnn_params,
    _sample_mlp_params,
    build_config,
)
from tests.conftest import REPO_ROOT

CONFIG_FILES = sorted((REPO_ROOT / "configs").glob("*.yaml"))


def _forward_shape(config: ExperimentConfig) -> tuple[int, ...]:
    return tuple(build_model(config)(torch.rand(2, 1, 28, 28)).shape)


# --- YAML config files ------------------------------------------------------


def test_config_files_are_found():
    # Guards the parametrized test below: if the glob matched nothing, it
    # would generate zero tests and the suite would pass while testing nothing.
    assert len(CONFIG_FILES) >= 4


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_file_loads_and_builds_model(path):
    config = ExperimentConfig.from_yaml(path)
    assert _forward_shape(config) == (2, 10)


def test_from_yaml_rejects_unknown_field(tmp_path):
    """A typo in a YAML key must fail loudly, not silently fall back to the default."""
    path = tmp_path / "typo.yaml"
    path.write_text("model:\n  dropuot: 0.5\n", encoding="utf-8")
    with pytest.raises(TypeError, match="dropuot"):
        ExperimentConfig.from_yaml(path)


def test_from_yaml_empty_file_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert ExperimentConfig.from_yaml(path) == ExperimentConfig()


@pytest.mark.parametrize(("architecture", "model_cls"), [("mlp", ModelConfig), ("cnn", CNNModelConfig)])
def test_from_yaml_architecture_selects_model_config(tmp_path, architecture, model_cls):
    path = tmp_path / "arch.yaml"
    path.write_text(f"architecture: {architecture}\n", encoding="utf-8")
    assert type(ExperimentConfig.from_yaml(path).model) is model_cls


# --- Optuna search space ----------------------------------------------------


def _valid_params(architecture: str, **overrides) -> dict:
    """One valid point in the search space; tests override the dimension they vary."""
    common = {
        "activation": "relu",
        "dropout": 0.1,
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "batch_size": 64,
    }
    if architecture == "cnn":
        specific = {"conv_channels": "baseline", "kernel_size": 3, "fc_hidden_dim": 128}
    else:
        specific = {"hidden_dims": "baseline"}
    return {**common, **specific, **overrides}


def _sample(architecture: str, params: dict) -> dict:
    """Run the real sampling function against fixed values. Optuna only *warns*
    when a float is outside its range, so warnings are promoted to errors.
    """
    sample_fn = _sample_cnn_params if architecture == "cnn" else _sample_mlp_params
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return sample_fn(optuna.trial.FixedTrial(params))


def _base(architecture: str) -> ExperimentConfig:
    model_config = CNNModelConfig() if architecture == "cnn" else ModelConfig()
    return ExperimentConfig(architecture=architecture, model=model_config)


@pytest.mark.parametrize("choice", list(_MLP_HIDDEN_DIM_CHOICES))
def test_every_mlp_search_choice_builds_a_model(choice):
    params = _sample("mlp", _valid_params("mlp", hidden_dims=choice))
    config = build_config(params, _base("mlp"))
    assert config.model.hidden_dims == _MLP_HIDDEN_DIM_CHOICES[choice]
    assert _forward_shape(config) == (2, 10)


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("choice", list(_CNN_CONV_CHANNELS_CHOICES))
def test_every_cnn_search_choice_builds_a_model(choice, kernel_size):
    params = _sample("cnn", _valid_params("cnn", conv_channels=choice, kernel_size=kernel_size))
    config = build_config(params, _base("cnn"))
    assert config.model.conv_channels == _CNN_CONV_CHANNELS_CHOICES[choice]
    assert _forward_shape(config) == (2, 10)


def test_build_config_keeps_base_settings_and_applies_batch_size():
    base = _base("mlp")
    base.data.split_seed = 123
    base.trainer.max_epochs = 7
    config = build_config(_sample("mlp", _valid_params("mlp", batch_size=256)), base)

    assert config.data.batch_size == 256
    # Everything not searched over must come from the base config -- above
    # all split_seed, or each trial would be scored on a different split.
    assert config.data.split_seed == 123
    assert config.data.data_dir == base.data.data_dir
    assert config.trainer.max_epochs == 7


def _to_search_params(config: ExperimentConfig) -> dict:
    """Inverse of build_config: express a config as raw search-space values."""
    model, optim = config.model, config.optim
    params = {
        "activation": model.activation,
        "dropout": model.dropout,
        "optimizer": optim.optimizer,
        "lr": optim.lr,
        "weight_decay": optim.weight_decay,
        "label_smoothing": optim.label_smoothing,
        "batch_size": config.data.batch_size,
    }
    if config.architecture == "cnn":
        names = [k for k, v in _CNN_CONV_CHANNELS_CHOICES.items() if v == model.conv_channels]
        params |= {"kernel_size": model.kernel_size, "fc_hidden_dim": model.fc_hidden_dim}
        params["conv_channels"] = names[0] if names else model.conv_channels
    else:
        names = [k for k, v in _MLP_HIDDEN_DIM_CHOICES.items() if v == model.hidden_dims]
        params["hidden_dims"] = names[0] if names else model.hidden_dims
    return params


@pytest.mark.parametrize("name", ["best_from_search.yaml", "cnn_best_from_search.yaml"])
def test_best_config_lies_inside_search_space(name):
    """A config labelled "best from search" must be a point the search could
    actually have produced -- and rebuilding it through build_config must give
    back the same model and optimizer settings.
    """
    config = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / name)
    params = _sample(config.architecture, _to_search_params(config))
    rebuilt = build_config(params, config)
    assert rebuilt.model == config.model
    assert rebuilt.optim == config.optim
    assert rebuilt.data == config.data
