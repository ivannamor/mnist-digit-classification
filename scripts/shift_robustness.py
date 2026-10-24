"""Regenerates the distribution-shift results: evaluates the committed MLP and
CNN checkpoints on clean and perturbed test images, prints the README table,
and saves results/shift_robustness.png.

Deterministic: run_shift_robustness seeds its noise, so the same checkpoints
and data always give the same numbers. Needs the real data in data/train.csv.

Usage:
    python scripts/shift_robustness.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig  # noqa: E402
from src.data_module import MNISTDataModule  # noqa: E402
from src.model import load_model  # noqa: E402
from src.robustness import run_shift_robustness  # noqa: E402

MODELS = {
    "MLP": ("configs/best_from_search.yaml", "results/mlp_best_from_search.ckpt"),
    "CNN": ("configs/cnn_best_from_search.yaml", "results/cnn_best_from_search.ckpt"),
}
LABELS = {
    "clean": "Clean",
    "rotate_15deg": "Rotate 15°",
    "rotate_30deg": "Rotate 30°",
    "translate_4px": "Translate 4px",
    "noise_sigma0.1": "Noise σ=0.1",  # noqa: RUF001 (Greek sigma, as in the README)
    "noise_sigma0.3": "Noise σ=0.3",  # noqa: RUF001
}
COLORS = {"MLP": "#4C72B0", "CNN": "#DD8452"}
OUTPUT = PROJECT_ROOT / "results" / "shift_robustness.png"


def compute() -> dict[str, dict[str, float]]:
    results = {}
    for name, (config_path, checkpoint) in MODELS.items():
        config = ExperimentConfig.from_yaml(PROJECT_ROOT / config_path)
        data_config = replace(config.data, data_dir=str(PROJECT_ROOT / config.data.data_dir), num_workers=0)
        data_module = MNISTDataModule(data_config)
        data_module.setup()
        model = load_model(config, str(PROJECT_ROOT / checkpoint))
        results[name] = run_shift_robustness(model, data_module.test_dataloader())
    return results


def plot(results: dict[str, dict[str, float]], output: Path) -> None:
    keys = list(LABELS)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, scores) in enumerate(results.items()):
        xs = [k + (i - 0.5) * width for k in range(len(keys))]
        values = [scores[key] * 100 for key in keys]
        bars = ax.bar(xs, values, width, label=name, color=COLORS[name])
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.axhline(10, color="gray", linestyle="--", linewidth=1, label="Chance level (10%)")
    ax.set_xticks(range(len(keys)), [LABELS[key] for key in keys], rotation=20, ha="right")
    ax.set_ylabel("Test accuracy, correct / total (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Distribution-shift robustness: accuracy on perturbed test images")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def main() -> None:
    # Windows consoles default to a legacy code page that can't print the
    # Greek sigma or the degree sign in the labels.
    sys.stdout.reconfigure(encoding="utf-8")
    results = compute()
    print("| Perturbation | " + " | ".join(results) + " |")
    print("|---|" + "---|" * len(results))
    for key, label in LABELS.items():
        print(f"| {label} | " + " | ".join(f"{scores[key]:.2%}" for scores in results.values()) + " |")
    plot(results, OUTPUT)
    print(f"Saved {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
