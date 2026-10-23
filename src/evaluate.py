from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch

from src.config import ExperimentConfig
from src.data_module import MNISTDataModule
from src.model import BaseClassifier, load_model

MisclassifiedExample = tuple[torch.Tensor, int, int]


def _collect_misclassified_examples(
    model: BaseClassifier, dataloader, max_examples: int = 16
) -> list[MisclassifiedExample]:
    model.eval()
    examples: list[MisclassifiedExample] = []
    with torch.no_grad():
        for images, labels in dataloader:
            preds = model(images).argmax(dim=-1)
            wrong = preds != labels
            for image, true_label, pred_label in zip(images[wrong], labels[wrong], preds[wrong], strict=True):
                examples.append((image, int(true_label), int(pred_label)))
                if len(examples) >= max_examples:
                    return examples
    return examples


def _plot_misclassified(examples: list[MisclassifiedExample], output_path: Path) -> None:
    if not examples:
        return
    cols = 4
    rows = (len(examples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    # axes.flat is longer than examples whenever len(examples) isn't a
    # multiple of cols -- the truncation is intentional (leftover axes are
    # turned off below), so no strict= here.
    for ax, (image, true_label, pred_label) in zip(axes.flat, examples):  # noqa: B905
        ax.imshow(image.squeeze(0), cmap="gray")
        ax.set_title(f"true={true_label} pred={pred_label}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(examples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)


def evaluate(config: ExperimentConfig, checkpoint_path: str, output_dir: Path) -> None:
    """The single, final pass over the test set — call this once, after the
    best configuration has already been chosen from validation results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data_module = MNISTDataModule(config.data)
    model = load_model(config, checkpoint_path)

    trainer = pl.Trainer(accelerator=config.trainer.accelerator, devices=config.trainer.devices, logger=False)
    trainer.test(model, datamodule=data_module)

    confusion_matrix = model.test_confusion_matrix.compute().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test set confusion matrix")
    fig.savefig(output_dir / "confusion_matrix.png")

    misclassified = _collect_misclassified_examples(model, data_module.test_dataloader())
    _plot_misclassified(misclassified, output_dir / "misclassified_examples.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    evaluate(config, str(args.checkpoint), args.output_dir)


if __name__ == "__main__":
    main()
