from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.config import CNNModelConfig, DataConfig, ExperimentConfig, ModelConfig, OptimConfig
from src.data_module import MNISTDataModule
from src.model import build_model

_MLP_HIDDEN_DIM_CHOICES = {
    "small": [256, 128],
    "baseline": [256, 128, 64],
    "wide": [512, 256],
}

_CNN_CONV_CHANNELS_CHOICES = {
    "narrow": [16, 32],
    "baseline": [32, 64],
    "wide": [64, 128],
}


def _sample_mlp_params(trial: optuna.Trial) -> dict:
    return {
        "hidden_dims": trial.suggest_categorical("hidden_dims", list(_MLP_HIDDEN_DIM_CHOICES)),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu", "tanh"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
        "label_smoothing": trial.suggest_categorical("label_smoothing", [0.0, 0.1]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def _sample_cnn_params(trial: optuna.Trial) -> dict:
    return {
        "conv_channels": trial.suggest_categorical("conv_channels", list(_CNN_CONV_CHANNELS_CHOICES)),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        "fc_hidden_dim": trial.suggest_categorical("fc_hidden_dim", [64, 128, 256]),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu", "tanh"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
        "label_smoothing": trial.suggest_categorical("label_smoothing", [0.0, 0.1]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def build_config(params: dict, base_config: ExperimentConfig) -> ExperimentConfig:
    """Builds an ExperimentConfig from a flat hyperparameter dict — either a
    trial's sampled params (during the search) or `study.best_params` (to
    retrain the winning configuration afterward). Dispatches on
    `base_config.architecture` so the same function serves both the MLP and
    CNN search spaces.
    """
    if base_config.architecture == "cnn":
        model_config = CNNModelConfig(
            conv_channels=_CNN_CONV_CHANNELS_CHOICES[params["conv_channels"]],
            kernel_size=params["kernel_size"],
            fc_hidden_dim=params["fc_hidden_dim"],
            activation=params["activation"],
            dropout=params["dropout"],
        )
    else:
        model_config = ModelConfig(
            hidden_dims=_MLP_HIDDEN_DIM_CHOICES[params["hidden_dims"]],
            activation=params["activation"],
            dropout=params["dropout"],
        )

    optim_config = OptimConfig(
        optimizer=params["optimizer"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        label_smoothing=params["label_smoothing"],
    )
    data_config = DataConfig(**{**base_config.data.__dict__, "batch_size": params["batch_size"]})
    return ExperimentConfig(
        architecture=base_config.architecture,
        data=data_config,
        model=model_config,
        optim=optim_config,
        trainer=base_config.trainer,
    )


def objective(trial: optuna.Trial, base_config: ExperimentConfig) -> float:
    sample_fn = _sample_cnn_params if base_config.architecture == "cnn" else _sample_mlp_params
    config = build_config(sample_fn(trial), base_config)
    pl.seed_everything(config.trainer.training_seed)

    data_module = MNISTDataModule(config.data)
    model = build_model(config)

    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        callbacks=[EarlyStopping(monitor="val_loss", patience=config.trainer.patience)],
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,  # trials only report a score; Lightning would otherwise save one per trial
    )
    trainer.fit(model, datamodule=data_module)
    return trainer.callback_metrics["val_accuracy"].item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    base_config = ExperimentConfig.from_yaml(args.config)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.n_trials)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
