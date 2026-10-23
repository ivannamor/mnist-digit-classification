"""Publishes src/ and configs/ as a private Kaggle Dataset, so a Kaggle
Notebook (which has no access to this local filesystem) can import the
project's code. Re-run this after any local change to src/ or configs/ that
should be reflected on Kaggle.

Kaggle only lets a dataset folder's subdirectories be uploaded zipped (each
becomes one archive, flattened at the zip root) -- see kaggle_train.ipynb in
notebooks/ for how those archives get unpacked back into an importable
`src` package inside the notebook.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Must run before importing kaggle: its __init__.py eagerly tries to
# authenticate at import time, so KAGGLE_API_TOKEN needs to already be in
# os.environ by then.
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
from requests.exceptions import HTTPError  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / ".kaggle_dataset_build"
DATASET_SLUG = "mnist-mlp-pipeline"
DATASET_TITLE = "MNIST MLP pipeline (src)"


def _stage_files() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)
    shutil.copytree(PROJECT_ROOT / "src", BUILD_DIR / "src")
    shutil.copytree(PROJECT_ROOT / "configs", BUILD_DIR / "configs")


def _write_metadata(owner_slug: str) -> str:
    ref = f"{owner_slug}/{DATASET_SLUG}"
    metadata = {"title": DATASET_TITLE, "id": ref, "licenses": [{"name": "CC0-1.0"}]}
    (BUILD_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    return ref


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    owner_slug = api.config_values["username"]

    _stage_files()
    ref = _write_metadata(owner_slug)

    try:
        api.dataset_status(ref)
        exists = True
    except HTTPError:
        exists = False

    if exists:
        response = api.dataset_create_version(str(BUILD_DIR), version_notes="Update from local project", dir_mode="zip")
    else:
        response = api.dataset_create_new(str(BUILD_DIR), public=False, dir_mode="zip")

    shutil.rmtree(BUILD_DIR)
    print(response)
    print(f"Dataset ref: {ref}")


if __name__ == "__main__":
    main()
