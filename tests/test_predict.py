from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from src.config import CNNModelConfig, ExperimentConfig
from src.model import build_model
from src.predict import main, predict, preprocess
from tests.conftest import REPO_ROOT


def _drawn_digit(size=(200, 200), box=(80, 40, 120, 160), background="white", ink="black", mode="L") -> Image.Image:
    """A thick vertical bar, roughly a '1', drawn like a user would: dark ink on light paper."""
    image = Image.new(mode, size, background)
    ImageDraw.Draw(image).rectangle(box, fill=ink)
    return image


def _center_of_mass(tensor: torch.Tensor) -> tuple[float, float]:
    pixels = tensor.squeeze(0).numpy()
    ys, xs = np.indices(pixels.shape)
    return (ys * pixels).sum() / pixels.sum(), (xs * pixels).sum() / pixels.sum()


# --- Preprocessing ----------------------------------------------------------


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(28, 28), (200, 120), (640, 480)])
def test_output_matches_training_format(mode, size):
    w, h = size
    image = _drawn_digit(size=size, box=(w // 3, h // 4, w // 2, 3 * h // 4), mode=mode)
    tensor = preprocess(image)
    assert tensor.shape == (1, 28, 28)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def _assert_mnist_polarity(tensor: torch.Tensor) -> None:
    # A thin digit covers a small part of the canvas. Checking one corner pixel
    # is not enough: if the page itself were taken as ink, it would be shrunk
    # into the central 20x20 box and the corners would still be black.
    assert tensor.max() > 0.9, "ink must become bright"
    assert tensor.mean() < 0.3, "background must become black"


def test_light_background_is_inverted_to_mnist_polarity():
    _assert_mnist_polarity(preprocess(_drawn_digit(background="white", ink="black")))


def test_dark_background_is_left_as_is():
    _assert_mnist_polarity(preprocess(_drawn_digit(background="black", ink="white")))


def test_transparent_background_counts_as_paper():
    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))  # fully transparent "black"
    ImageDraw.Draw(image).rectangle((80, 40, 120, 160), fill=(0, 0, 0, 255))
    _assert_mnist_polarity(preprocess(image))


@pytest.mark.parametrize("box", [(5, 5, 30, 60), (150, 120, 190, 195), (90, 80, 110, 120)])
def test_off_center_digit_is_centered(box):
    """The translation weakness from the robustness checks: position must not matter."""
    center_y, center_x = _center_of_mass(preprocess(_drawn_digit(box=box)))
    assert abs(center_y - 14) <= 1
    assert abs(center_x - 14) <= 1


def test_digit_scaled_to_fit_20px_box_keeping_aspect_ratio():
    tensor = preprocess(_drawn_digit(box=(80, 40, 119, 159)))  # 40 wide x 120 tall -> 3:1
    rows, cols = np.nonzero(tensor.squeeze(0).numpy() > 0.5)
    height, width = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    assert height == 20
    assert 5 <= width <= 8  # 20 / 3, allowing for resampling at the edges


def test_same_digit_at_different_positions_gives_same_tensor():
    a = preprocess(_drawn_digit(box=(10, 10, 50, 130)))
    b = preprocess(_drawn_digit(box=(140, 60, 180, 180)))
    torch.testing.assert_close(a, b)


def test_blank_image_raises():
    with pytest.raises(ValueError, match="no digit"):
        preprocess(Image.new("L", (100, 100), "white"))


# --- Prediction -------------------------------------------------------------


def test_predict_returns_probabilities():
    model = build_model(ExperimentConfig(architecture="cnn", model=CNNModelConfig()))
    probabilities = predict(model, _drawn_digit())
    assert probabilities.shape == (10,)
    assert torch.all(probabilities >= 0)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))


def test_cli_prints_prediction_per_image(tmp_path, monkeypatch, capsys):
    checkpoint = REPO_ROOT / "results" / "cnn_best_from_search.ckpt"
    if not checkpoint.exists():
        pytest.skip("committed CNN checkpoint not present")
    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in paths:
        _drawn_digit().save(path)
    monkeypatch.chdir(REPO_ROOT)  # the CLI's default config/checkpoint paths are repo-relative
    monkeypatch.setattr(sys, "argv", ["predict", *map(str, paths)])

    main()

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    for path, line in zip(paths, lines, strict=True):
        assert line.startswith(f"{path}: ")
        assert "%" in line
