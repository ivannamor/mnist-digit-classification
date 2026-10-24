"""Shared fixtures. Tests never touch the real Kaggle data -- they build a
small synthetic CSV in the same format as Kaggle's train.csv, so the suite
runs offline, on CPU, in CI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DataConfig
from src.data_module import _load_clean_and_split

NUM_CLASSES = 10
ROWS_PER_CLASS = 50


def make_mnist_frame(rows_per_class: int = ROWS_PER_CLASS, seed: int = 0) -> pd.DataFrame:
    """Kaggle train.csv layout: a `label` column, then `pixel0`..`pixel783`.
    Pixels are random in [1, 255] so no row is blank and no two rows collide.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(NUM_CLASSES), rows_per_class)
    pixels = rng.integers(1, 256, size=(len(labels), 28 * 28))
    df = pd.DataFrame(pixels, columns=[f"pixel{i}" for i in range(28 * 28)])
    df.insert(0, "label", labels)
    return df


@pytest.fixture
def mnist_csv_dir(tmp_path: Path) -> Path:
    make_mnist_frame().to_csv(tmp_path / "train.csv", index=False)
    return tmp_path


@pytest.fixture
def data_config(mnist_csv_dir: Path) -> DataConfig:
    # num_workers=0: worker processes add seconds of startup on Windows for no benefit at this size.
    return DataConfig(data_dir=str(mnist_csv_dir), batch_size=16, num_workers=0)


@pytest.fixture(autouse=True)
def _clear_split_cache():
    # _load_clean_and_split is functools.cache'd for the whole process; clear
    # it so one test's split can never leak into another.
    _load_clean_and_split.cache_clear()
    yield
    _load_clean_and_split.cache_clear()
