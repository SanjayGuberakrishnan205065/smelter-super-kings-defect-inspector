"""Shared constants and file helpers for the image-track capstone."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CLASS_NAMES = ["acceptable", "scratch", "dent", "stain", "other"]
PRIORITY_CLASS = "stain"
PRODUCT_TYPES = ["coil", "sheet", "plate", "strip"]
DEFAULT_IMAGE_SIZE = (160, 160)
DEFAULT_IMAGES_PER_CLASS = 40
DEFAULT_BASELINE_IMAGE_SIZE = (32, 32)
DEFAULT_TRAIN_TEST_SEED = 42
DEFAULT_TEST_RATIO = 0.2
DATASET_NOTICE_FILENAME = "SYNTHETIC_DATASET_NOTICE.txt"
METADATA_FILENAME = "metadata.csv"


def get_class_names() -> list[str]:
    """Return the ordered class list used across training and inference."""
    return list(CLASS_NAMES)


def get_priority_class() -> str:
    """Return the priority defect class for evaluation and reporting."""
    return PRIORITY_CLASS


def get_product_types() -> list[str]:
    """Return the allowed product types for synthetic metadata generation."""
    return list(PRODUCT_TYPES)


def get_dataset_notice_path(data_dir: str | Path) -> Path:
    """Return the path to the dataset-level synthetic notice file."""
    return Path(data_dir) / DATASET_NOTICE_FILENAME


def get_metadata_path(data_dir: str | Path) -> Path:
    """Return the path to the dataset metadata file."""
    return Path(data_dir) / METADATA_FILENAME


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionary rows."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv_rows(path: str | Path, rows: list[dict[str, str]], fieldnames: list[str]) -> Path:
    """Write dictionary rows to a CSV file."""
    csv_path = Path(path)
    ensure_directory(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload to disk with readable formatting."""
    json_path = Path(path)
    ensure_directory(json_path.parent)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON document from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
