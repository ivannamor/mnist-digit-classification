from __future__ import annotations

import zipfile
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig

load_dotenv()


def _load_and_clean(data_dir: str, train_csv: str) -> pd.DataFrame:
    train_path = Path(data_dir) / train_csv
    df = pd.read_csv(train_path)

    df = df.dropna()
    df = df.drop_duplicates()

    pixel_cols = [c for c in df.columns if c != "label"]
    valid_pixels = df[pixel_cols].apply(lambda col: col.between(0, 255)).all(axis=1)
    valid_labels = df["label"].between(0, 9)
    df = df[valid_pixels & valid_labels]

    is_blank = df[pixel_cols].sum(axis=1) == 0
    df = df[~is_blank]

    return df.reset_index(drop=True)


@cache
def _load_clean_and_split(
    data_dir: str, train_csv: str, split_seed: int, val_fraction: float, test_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cached so that many `MNISTDataModule`s built from the same parameters
    (e.g. one per Optuna trial, or one per cross-seed retrain -- all within
    the same process) reuse the already-loaded/cleaned/split arrays instead
    of re-parsing the 76MB CSV from scratch every time. Safe to share: the
    returned arrays are never mutated by any caller.
    """
    df = _load_and_clean(data_dir, train_csv)
    pixel_cols = [c for c in df.columns if c != "label"]
    pixels = df[pixel_cols].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)

    val_test_fraction = val_fraction + test_fraction
    pixels_train, pixels_val_test, labels_train, labels_val_test = train_test_split(
        pixels,
        labels,
        test_size=val_test_fraction,
        stratify=labels,
        random_state=split_seed,
    )
    test_ratio_within_val_test = test_fraction / val_test_fraction
    pixels_val, pixels_test, labels_val, labels_test = train_test_split(
        pixels_val_test,
        labels_val_test,
        test_size=test_ratio_within_val_test,
        stratify=labels_val_test,
        random_state=split_seed,
    )
    return pixels_train, labels_train, pixels_val, labels_val, pixels_test, labels_test


class MNISTCsvDataset(Dataset):
    def __init__(self, pixels: np.ndarray, labels: np.ndarray):
        self.pixels = pixels
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = torch.from_numpy(self.pixels[idx]).float().view(1, 28, 28) / 255.0
        label = int(self.labels[idx])
        return image, label


class MNISTDataModule(pl.LightningDataModule):
    """Loads Kaggle's Digit Recognizer train.csv and builds the 60/20/20 split.

    Kaggle's competition test.csv has no labels (leaderboard-submission only),
    so it is intentionally not used here — the full split comes from train.csv.
    """

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.train_dataset: MNISTCsvDataset | None = None
        self.val_dataset: MNISTCsvDataset | None = None
        self.test_dataset: MNISTCsvDataset | None = None

    def prepare_data(self) -> None:
        train_path = Path(self.config.data_dir) / self.config.train_csv
        if train_path.exists():
            return

        # Deferred: importing `kaggle` triggers an eager, noisy auth attempt
        # at import time (its __init__.py does this), so only pay for it
        # when a download is actually about to happen.
        from kaggle.api.kaggle_api_extended import KaggleApi

        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("digit-recognizer", path=self.config.data_dir)

        zip_path = Path(self.config.data_dir) / "digit-recognizer.zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.config.data_dir)

    def setup(self, stage: str | None = None) -> None:
        pixels_train, labels_train, pixels_val, labels_val, pixels_test, labels_test = _load_clean_and_split(
            self.config.data_dir,
            self.config.train_csv,
            self.config.split_seed,
            self.config.val_fraction,
            self.config.test_fraction,
        )

        self.train_dataset = MNISTCsvDataset(pixels_train, labels_train)
        self.val_dataset = MNISTCsvDataset(pixels_val, labels_val)
        self.test_dataset = MNISTCsvDataset(pixels_test, labels_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
