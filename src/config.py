from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data"
    train_csv: str = "train.csv"
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    batch_size: int = 64
    num_workers: int = 2
    # Controls only the train/val/test split. Kept fixed across cross-seed
    # trials so every trial is evaluated on the same held-out test set --
    # see TrainerConfig.training_seed for the model-init/shuffling seed.
    split_seed: int = 42
    # Data augmentation, applied to training images only (val/test stay
    # untouched so results remain comparable): every time an image is loaded
    # it is shifted by up to this many pixels per axis and rotated by up to
    # this many degrees, at random. 0 = off, the default.
    augment_max_shift_px: int = 0
    augment_max_rotation_deg: float = 0.0


@dataclass
class ModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "relu"
    dropout: float = 0.25
    num_classes: int = 10


@dataclass
class CNNModelConfig:
    conv_channels: list[int] = field(default_factory=lambda: [32, 64])
    kernel_size: int = 3
    activation: str = "relu"
    dropout: float = 0.25
    fc_hidden_dim: int = 128
    num_classes: int = 10


@dataclass
class OptimConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    lr_scheduler: str | None = None


@dataclass
class TrainerConfig:
    max_epochs: int = 30
    accelerator: str = "auto"
    devices: int = 1
    patience: int = 5
    experiment_name: str = "baseline"
    # Seeds model init / dataloader shuffling / dropout masks (pl.seed_everything).
    # Deliberately separate from DataConfig.split_seed so cross-seed trials can
    # vary this while every trial still trains/evaluates on the same split.
    training_seed: int = 42


@dataclass
class ExperimentConfig:
    architecture: str = "mlp"  # "mlp" or "cnn" -- selects the model dataclass and LightningModule
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig | CNNModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        architecture = raw.get("architecture", "mlp")
        model_cls = CNNModelConfig if architecture == "cnn" else ModelConfig
        return cls(
            architecture=architecture,
            data=DataConfig(**raw.get("data", {})),
            model=model_cls(**raw.get("model", {})),
            optim=OptimConfig(**raw.get("optim", {})),
            trainer=TrainerConfig(**raw.get("trainer", {})),
        )
