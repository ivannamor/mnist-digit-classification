"""Regenerates every distribution-shift result in the README from the
committed checkpoints, for the original and the augmented MLP and CNN:

- the perturbation table (rotation, translation, noise) and its plots:
  results/shift_robustness.png (original models) and
  results/augmentation_shift_robustness.png (original vs augmented);
- the translation curve (accuracy vs shift of 0-8 px):
  results/translation_curve.png;
- the noise-location check (noise everywhere / border only / centre only).

Deterministic: noise is seeded, so the same checkpoints and data always give
the same numbers. Needs the real data in data/train.csv.

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
from src.robustness import run_noise_location_check, run_shift_robustness, run_translation_curve  # noqa: E402

MODELS = {
    "MLP": ("configs/best_from_search.yaml", "results/mlp_best_from_search.ckpt"),
    "MLP augmented": ("configs/augmented.yaml", "results/augmented.ckpt"),
    "CNN": ("configs/cnn_best_from_search.yaml", "results/cnn_best_from_search.ckpt"),
    "CNN augmented": ("configs/cnn_augmented.yaml", "results/cnn_augmented.ckpt"),
}
LABELS = {
    "clean": "Clean",
    "rotate_15deg": "Rotate 15°",
    "rotate_30deg": "Rotate 30°",
    "translate_4px": "Translate 4px",
    "noise_sigma0.1": "Noise σ=0.1",  # noqa: RUF001 (Greek sigma, as in the README)
    "noise_sigma0.3": "Noise σ=0.3",  # noqa: RUF001
}
NOISE_LABELS = {"everywhere": "Noise everywhere", "border_only": "Border only", "center_only": "Centre only"}
COLORS = {"MLP": "#4C72B0", "CNN": "#DD8452"}
TRAINED_SHIFT_PX = 4  # augment_max_shift_px in configs/*augmented.yaml


def _style(name: str) -> dict:
    """Same colour per architecture; augmented variants are hatched (white hatch lines)."""
    style = {"color": COLORS[name.split()[0]]}
    if "augmented" in name:
        style |= {"hatch": "//", "edgecolor": "white"}
    return style


def compute() -> dict[str, dict]:
    results = {}
    for name, (config_path, checkpoint) in MODELS.items():
        config = ExperimentConfig.from_yaml(PROJECT_ROOT / config_path)
        data_config = replace(config.data, data_dir=str(PROJECT_ROOT / config.data.data_dir), num_workers=0)
        data_module = MNISTDataModule(data_config)
        data_module.setup()
        model = load_model(config, str(PROJECT_ROOT / checkpoint))
        loader = data_module.test_dataloader()
        results[name] = {
            "perturbations": run_shift_robustness(model, loader),
            "translation": run_translation_curve(model, loader),
            "noise_location": run_noise_location_check(model, loader),
        }
    return results


def plot_bars(scores: dict[str, dict[str, float]], output: Path) -> None:
    keys = list(LABELS)
    width = 0.72 / len(scores)  # 0.36 for the two original models, as in the published figure
    fig, ax = plt.subplots(figsize=(11 if len(scores) > 2 else 10, 6))
    for i, (name, values) in enumerate(scores.items()):
        xs = [k + (i - (len(scores) - 1) / 2) * width for k in range(len(keys))]
        bars = ax.bar(xs, [values[key] * 100 for key in keys], width, label=name, **_style(name))
        ax.bar_label(bars, fmt="%.0f" if len(scores) > 2 else "%.1f", padding=2, fontsize=7 if len(scores) > 2 else 8)
    ax.axhline(10, color="gray", linestyle="--", linewidth=1, label="Chance level (10%)")
    ax.set_xticks(range(len(keys)), [LABELS[key] for key in keys], rotation=20, ha="right")
    ax.set_ylabel("Test accuracy, correct / total (%)")
    ax.set_ylim(0, 105)
    title = "original vs augmented models" if len(scores) > 2 else "accuracy on perturbed test images"
    ax.set_title(f"Distribution-shift robustness: {title}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(scores) + 1, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_translation_curve(curves: dict[str, dict[int, float]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(0, TRAINED_SHIFT_PX, color="gray", alpha=0.12, label=f"Shifts seen in training (≤{TRAINED_SHIFT_PX} px)")
    for name, curve in curves.items():
        style = _style(name)
        ax.plot(
            list(curve),
            [v * 100 for v in curve.values()],
            marker="o",
            color=style["color"],
            linestyle="-" if "augmented" in name else "--",
            label=name,
        )
    ax.axhline(10, color="gray", linestyle=":", linewidth=1, label="Chance level (10%)")
    ax.set_xlabel("Shift of every test image, right and down (px)")
    ax.set_ylabel("Test accuracy, correct / total (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(range(len(next(iter(curves.values())))))
    ax.set_title("Accuracy vs shift: augmentation extends robustness beyond the trained range")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _print_table(header: str, columns: list[str], rows: dict[str, list[str]]) -> None:
    print(f"\n| {header} | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for label, cells in rows.items():
        print(f"| {label} | " + " | ".join(cells) + " |")


def main() -> None:
    # Windows consoles default to a legacy code page that can't print the
    # Greek sigma or the degree sign in the labels.
    sys.stdout.reconfigure(encoding="utf-8")
    results = compute()
    names = list(results)

    _print_table(
        "Perturbation",
        names,
        {label: [f"{results[n]['perturbations'][key]:.2%}" for n in names] for key, label in LABELS.items()},
    )
    shifts = list(results[names[0]]["translation"])
    _print_table(
        "Shift (px)",
        [str(s) for s in shifts],
        {n: [f"{results[n]['translation'][s]:.1%}" for s in shifts] for n in names},
    )
    _print_table(
        "Noise σ=0.3",  # noqa: RUF001
        list(NOISE_LABELS.values()),
        {n: [f"{results[n]['noise_location'][key]:.1%}" for key in NOISE_LABELS] for n in names},
    )

    perturbations = {n: results[n]["perturbations"] for n in names}
    originals = {n: scores for n, scores in perturbations.items() if "augmented" not in n}
    curves = {n: results[n]["translation"] for n in names}
    outputs = {
        "shift_robustness.png": lambda path: plot_bars(originals, path),
        "augmentation_shift_robustness.png": lambda path: plot_bars(perturbations, path),
        "translation_curve.png": lambda path: plot_translation_curve(curves, path),
    }
    for filename, draw in outputs.items():
        draw(PROJECT_ROOT / "results" / filename)
        print(f"Saved results/{filename}")


if __name__ == "__main__":
    main()
