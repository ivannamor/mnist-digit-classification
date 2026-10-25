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


class _PruningCallback(pl.Callback):
    """Reports val_accuracy to Optuna after every validation epoch and stops
    the trial once the pruner judges it clearly worse than earlier trials at
    the same epoch. Without these reports the study's pruner has nothing to
    compare and every trial runs to the end. (Optuna's own Lightning callback
    lives in the separate optuna-integration package; this is the same idea.)
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_accuracy"):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:  # Lightning's pre-training check of 2 val batches
            return
        epoch = trainer.current_epoch
        self.trial.report(trainer.callback_metrics[self.monitor].item(), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned after epoch {epoch}")


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
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config.trainer.patience),
            _PruningCallback(trial),
        ],
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,  # trials only report a score; Lightning would otherwise save one per trial
    )
    trainer.fit(model, datamodule=data_module)
    return trainer.callback_metrics["val_accuracy"].item()


def create_study() -> optuna.Study:
    # A trial can only be pruned after 5 complete trials exist to compare
    # against, and not during its first 3 epochs, when scores are still noisy.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    return optuna.create_study(direction="maximize", pruner=pruner)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    base_config = ExperimentConfig.from_yaml(args.config)
    study = create_study()
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.n_trials)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
