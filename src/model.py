from __future__ import annotations

from dataclasses import asdict
from itertools import pairwise

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.config import CNNModelConfig, ExperimentConfig, ModelConfig, OptimConfig

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
_OPTIMIZERS = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


class BaseClassifier(pl.LightningModule):
    """Shared training/eval logic for all architectures. Subclasses implement
    `_build_net` (the model-specific layers) and `forward` (whether the input
    needs flattening before the net, which differs between MLP and CNN).
    """

    def __init__(self, model_config, optim_config: OptimConfig):
        super().__init__()
        self.model_config = model_config
        self.optim_config = optim_config

        self.net = self._build_net(model_config)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=optim_config.label_smoothing)

        num_classes = model_config.num_classes
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

    def _build_net(self, model_config) -> nn.Sequential:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=-1)
        self.val_accuracy.update(preds, y)
        self.val_precision.update(preds, y)
        self.val_recall.update(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=-1)
        self.test_accuracy.update(preds, y)
        self.test_precision.update(preds, y)
        self.test_recall.update(preds, y)
        self.test_confusion_matrix.update(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = _OPTIMIZERS[self.optim_config.optimizer](
            self.parameters(),
            lr=self.optim_config.lr,
            weight_decay=self.optim_config.weight_decay,
        )

        if self.optim_config.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.optim_config.lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        if self.optim_config.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

        return optimizer


class MLPClassifier(BaseClassifier):
    """Configurable MLP. Flattening happens here (not in the data module) so
    CNNClassifier can reuse MNISTDataModule's (1, 28, 28) tensors unchanged.
    """

    def __init__(self, model_config: ModelConfig | dict, optim_config: OptimConfig | dict):
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        if isinstance(optim_config, dict):
            optim_config = OptimConfig(**optim_config)
        super().__init__(model_config, optim_config)
        # Stored as plain dicts (not dataclass instances) so checkpoints stay
        # loadable under torch.load's weights_only=True default (PyTorch 2.6+).
        self.save_hyperparameters({"model_config": asdict(model_config), "optim_config": asdict(optim_config)})

    @staticmethod
    def _build_net(model_config: ModelConfig) -> nn.Sequential:
        activation_cls = _ACTIVATIONS[model_config.activation]
        dims: list[int] = [28 * 28, *model_config.hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in pairwise(dims):
            layers += [nn.Linear(in_dim, out_dim), activation_cls(), nn.Dropout(model_config.dropout)]
        layers.append(nn.Linear(dims[-1], model_config.num_classes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


class CNNClassifier(BaseClassifier):
    """Small CNN: `len(conv_channels)` conv+pool blocks, then a fully
    connected head. Consumes MNISTDataModule's (1, 28, 28) tensors directly --
    no flattening before the conv layers, unlike the MLP.
    """

    def __init__(self, model_config: CNNModelConfig | dict, optim_config: OptimConfig | dict):
        if isinstance(model_config, dict):
            model_config = CNNModelConfig(**model_config)
        if isinstance(optim_config, dict):
            optim_config = OptimConfig(**optim_config)
        super().__init__(model_config, optim_config)
        self.save_hyperparameters({"model_config": asdict(model_config), "optim_config": asdict(optim_config)})

    @staticmethod
    def _build_net(model_config: CNNModelConfig) -> nn.Sequential:
        activation_cls = _ACTIVATIONS[model_config.activation]
        layers: list[nn.Module] = []
        in_channels = 1
        spatial_size = 28
        for out_channels in model_config.conv_channels:
            layers += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=model_config.kernel_size,
                    padding=model_config.kernel_size // 2,
                ),
                activation_cls(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
            spatial_size //= 2

        flat_dim = in_channels * spatial_size * spatial_size
        layers += [
            nn.Flatten(),
            nn.Linear(flat_dim, model_config.fc_hidden_dim),
            activation_cls(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.fc_hidden_dim, model_config.num_classes),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_MODEL_CLASSES = {"mlp": MLPClassifier, "cnn": CNNClassifier}


def build_model(config: ExperimentConfig) -> BaseClassifier:
    return _MODEL_CLASSES[config.architecture](config.model, config.optim)


def load_model(config: ExperimentConfig, checkpoint_path: str) -> BaseClassifier:
    return _MODEL_CLASSES[config.architecture].load_from_checkpoint(checkpoint_path)
