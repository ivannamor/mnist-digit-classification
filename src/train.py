from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.config import ExperimentConfig
from src.data_module import MNISTDataModule
from src.model import build_model


def train(config: ExperimentConfig) -> pl.Trainer:
    """Fits and validates only. The test set is deliberately not touched here
    — see evaluate.py, which runs the single, final test-set pass.
    """
    pl.seed_everything(config.trainer.training_seed)

    data_module = MNISTDataModule(config.data)
    model = build_model(config)

    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max", save_top_k=1, filename="best")
    callbacks = [
        checkpoint_callback,
        EarlyStopping(monitor="val_loss", patience=config.trainer.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = CSVLogger("outputs/logs", name=config.trainer.experiment_name)

    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model, datamodule=data_module)

    print(f"Best val_accuracy: {checkpoint_callback.best_model_score}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
