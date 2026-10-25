from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from src.config import DataConfig
from src.data_module import MNISTDataModule, _load_and_clean, _load_clean_and_split, build_train_transform
from tests.conftest import NUM_CLASSES, ROWS_PER_CLASS, make_mnist_frame


def _setup(config) -> MNISTDataModule:
    data_module = MNISTDataModule(config)
    data_module.setup()
    return data_module


# --- Cleaning ---------------------------------------------------------------


def test_cleaning_removes_each_kind_of_bad_row(tmp_path):
    clean = make_mnist_frame()
    template = clean.iloc[0].copy()

    duplicate = clean.iloc[[1]]
    with_nan = template.to_frame().T.assign(pixel5=np.nan)
    pixel_too_high = template.to_frame().T.assign(pixel5=300)
    pixel_negative = template.to_frame().T.assign(pixel5=-1)
    label_out_of_range = template.to_frame().T.assign(label=11)
    blank = template.to_frame().T.assign(**{f"pixel{i}": 0 for i in range(28 * 28)})

    dirty = pd.concat(
        [clean, duplicate, with_nan, pixel_too_high, pixel_negative, label_out_of_range, blank],
        ignore_index=True,
    )
    dirty.to_csv(tmp_path / "train.csv", index=False)

    result = _load_and_clean(str(tmp_path), "train.csv")

    assert len(result) == len(clean)
    pd.testing.assert_frame_equal(result, clean, check_dtype=False)


def test_cleaning_keeps_already_clean_data_unchanged(mnist_csv_dir):
    result = _load_and_clean(str(mnist_csv_dir), "train.csv")
    assert len(result) == NUM_CLASSES * ROWS_PER_CLASS


# --- Split ------------------------------------------------------------------


def test_split_sizes_match_configured_fractions(data_config):
    dm = _setup(data_config)
    total = NUM_CLASSES * ROWS_PER_CLASS
    assert len(dm.train_dataset) == total * 0.6
    assert len(dm.val_dataset) == total * 0.2
    assert len(dm.test_dataset) == total * 0.2


def test_splits_do_not_overlap(data_config):
    """Leakage check: no image may appear in more than one partition."""
    dm = _setup(data_config)
    as_rows = [
        {row.tobytes() for row in split.pixels} for split in (dm.train_dataset, dm.val_dataset, dm.test_dataset)
    ]
    train, val, test = as_rows
    assert not train & val
    assert not train & test
    assert not val & test
    assert len(train | val | test) == NUM_CLASSES * ROWS_PER_CLASS


def test_split_is_stratified(data_config):
    dm = _setup(data_config)
    for split, expected_per_class in [
        (dm.train_dataset, ROWS_PER_CLASS * 0.6),
        (dm.val_dataset, ROWS_PER_CLASS * 0.2),
        (dm.test_dataset, ROWS_PER_CLASS * 0.2),
    ]:
        counts = Counter(split.labels.tolist())
        assert set(counts) == set(range(NUM_CLASSES))
        assert all(count == expected_per_class for count in counts.values())


def test_split_is_deterministic_for_same_seed(data_config):
    first = _setup(data_config)
    _load_clean_and_split.cache_clear()  # force a genuine recompute, not a cache hit
    second = _setup(data_config)
    np.testing.assert_array_equal(first.test_dataset.pixels, second.test_dataset.pixels)
    np.testing.assert_array_equal(first.test_dataset.labels, second.test_dataset.labels)


def test_split_changes_with_different_seed(data_config):
    first = _setup(data_config)
    second = _setup(replace(data_config, split_seed=data_config.split_seed + 1))
    assert not np.array_equal(first.test_dataset.pixels, second.test_dataset.pixels)


# --- Tensors fed to the model ----------------------------------------------


def test_dataset_item_shape_dtype_and_range(data_config):
    image, label = _setup(data_config).train_dataset[0]
    assert image.shape == (1, 28, 28)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert isinstance(label, int)
    assert 0 <= label < NUM_CLASSES


def test_dataloader_batches(data_config):
    images, labels = next(iter(_setup(data_config).train_dataloader()))
    assert images.shape == (data_config.batch_size, 1, 28, 28)
    assert labels.shape == (data_config.batch_size,)


def test_prepare_data_skips_download_when_csv_exists(data_config, monkeypatch):
    # A None entry in sys.modules makes any `import kaggle...` raise ImportError,
    # so this fails loudly if prepare_data reaches the download branch.
    monkeypatch.setitem(sys.modules, "kaggle", None)
    MNISTDataModule(data_config).prepare_data()


# --- Augmentation -----------------------------------------------------------


def _augmented(config):
    return replace(config, augment_max_shift_px=4, augment_max_rotation_deg=15)


def test_no_augmentation_by_default(data_config):
    """Existing configs and checkpoints must behave exactly as before."""
    dm = _setup(data_config)
    assert dm.train_dataset.transform is None
    torch.testing.assert_close(dm.train_dataset[0][0], dm.train_dataset[0][0])


def test_augmentation_alters_training_images_on_every_load(data_config):
    dm = _setup(_augmented(data_config))
    torch.manual_seed(0)
    loads = [dm.train_dataset[0][0] for _ in range(5)]
    assert any(not torch.equal(loads[0], other) for other in loads[1:])
    for image in loads:
        assert image.shape == (1, 28, 28)
        assert 0.0 <= image.min() and image.max() <= 1.0
    assert dm.train_dataset[0][1] == dm.train_dataset[0][1], "the label must never change"


def test_augmentation_never_touches_val_or_test(data_config):
    """Val/test stay clean, so augmented and original models are scored on identical images."""
    plain, augmented = _setup(data_config), _setup(_augmented(data_config))
    for split in ("val_dataset", "test_dataset"):
        for idx in range(3):
            torch.testing.assert_close(getattr(augmented, split)[idx][0], getattr(plain, split)[idx][0])
        assert getattr(augmented, split).transform is None


def test_shift_stays_within_configured_pixels():
    transform = build_train_transform(DataConfig(augment_max_shift_px=4))
    image = torch.zeros(1, 28, 28)
    image[0, 14, 14] = 1.0
    torch.manual_seed(0)
    shifts = set()
    for _ in range(500):
        row, col = divmod(int(transform(image).flatten().argmax()), 28)
        shifts.add((row - 14, col - 14))
    assert max(max(abs(dy), abs(dx)) for dy, dx in shifts) == 4  # reaches the limit...
    assert all(abs(dy) <= 4 and abs(dx) <= 4 for dy, dx in shifts)  # ...and never exceeds it
