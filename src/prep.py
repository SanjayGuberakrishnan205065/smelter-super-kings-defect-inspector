"""Dataset validation utilities for the image-track capstone."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image
from sklearn.model_selection import train_test_split

from src.model_utils import (
    CLASS_NAMES,
    DATASET_NOTICE_FILENAME,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_TEST_SEED,
    METADATA_FILENAME,
    get_class_names,
    get_metadata_path,
    write_csv_rows,
)


REQUIRED_METADATA_COLUMNS = {
    "sample_id",
    "label",
    "product_type",
    "image_path",
    "is_synthetic",
    "synthetic_note",
    "width",
    "height",
}


def load_metadata_rows(data_dir: str | Path) -> list[dict[str, str]]:
    """Load dataset metadata rows from the generated CSV manifest."""
    metadata_path = get_metadata_path(data_dir)
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def validate_generated_dataset(
    data_dir: str | Path,
    min_images_per_class: int = 1,
    expected_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, Any]:
    """Validate the generated dataset structure and return a report."""
    data_path = Path(data_dir)
    expected_classes = get_class_names()
    errors: list[str] = []
    summary: dict[str, Any] = {
        "data_dir": str(data_path),
        "expected_classes": expected_classes,
        "class_counts": {},
        "checked_images": 0,
        "errors": errors,
    }

    if not data_path.exists():
        errors.append(f"Dataset directory does not exist: {data_path}")
        return summary

    notice_path = data_path / DATASET_NOTICE_FILENAME
    if not notice_path.exists():
        errors.append(f"Missing synthetic dataset notice file: {notice_path}")
    else:
        notice_text = notice_path.read_text(encoding="utf-8")
        if "SYNTHETIC" not in notice_text.upper():
            errors.append("Synthetic dataset notice file does not clearly mark the data as SYNTHETIC.")

    metadata_path = data_path / METADATA_FILENAME
    if not metadata_path.exists():
        errors.append(f"Missing metadata file: {metadata_path}")
        return summary

    rows = load_metadata_rows(data_path)
    if not rows:
        errors.append("Metadata file exists but contains no rows.")
        return summary

    header_columns = set(rows[0].keys())
    missing_columns = REQUIRED_METADATA_COLUMNS - header_columns
    if missing_columns:
        errors.append(f"Metadata file is missing required columns: {sorted(missing_columns)}")

    class_counts = Counter()

    for row in rows:
        label = row.get("label", "")
        image_path_value = row.get("image_path", "")

        if label not in expected_classes:
            errors.append(f"Unexpected label in metadata: {label}")
            continue

        class_counts[label] += 1
        image_path = data_path / image_path_value

        if row.get("is_synthetic", "").strip().lower() != "true":
            errors.append(f"Row for {image_path_value} is not marked synthetic.")

        if "synthetic" not in row.get("synthetic_note", "").strip().lower():
            errors.append(f"Row for {image_path_value} does not contain a clear synthetic note.")

        if not image_path.exists():
            errors.append(f"Missing image referenced by metadata: {image_path}")
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.size != expected_size:
                    errors.append(
                        f"Unexpected image size for {image_path_value}: {image.size}, expected {expected_size}"
                    )
        except Exception as exc:  # pragma: no cover - exact PIL failure varies
            errors.append(f"Unreadable image {image_path}: {exc}")
            continue

        width = row.get("width", "")
        height = row.get("height", "")
        if width != str(expected_size[0]) or height != str(expected_size[1]):
            errors.append(
                f"Metadata dimensions do not match expected size for {image_path_value}: "
                f"({width}, {height}) vs {expected_size}"
            )

        summary["checked_images"] += 1

    summary["class_counts"] = dict(class_counts)

    for class_name in expected_classes:
        class_dir = data_path / class_name
        if not class_dir.exists():
            errors.append(f"Missing class directory: {class_dir}")
            continue

        if class_counts[class_name] < min_images_per_class:
            errors.append(
                f"Class '{class_name}' has {class_counts[class_name]} images, "
                f"expected at least {min_images_per_class}."
            )

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset validation."""
    parser = argparse.ArgumentParser(description="Validate the generated synthetic image dataset.")
    parser.add_argument("--data_dir", default="data/images", help="Directory containing the generated dataset.")
    parser.add_argument(
        "--min_images_per_class",
        type=int,
        default=1,
        help="Minimum number of images expected in each class directory.",
    )
    parser.add_argument("--train_out", help="Optional CSV path for the training split metadata.")
    parser.add_argument("--test_out", help="Optional CSV path for the test split metadata.")
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Proportion of rows reserved for the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAIN_TEST_SEED,
        help="Random seed for deterministic train/test splitting.",
    )
    return parser.parse_args()


def write_split_metadata(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    """Write split metadata rows to a CSV file."""
    fieldnames = list(rows[0].keys()) if rows else sorted(REQUIRED_METADATA_COLUMNS)
    return write_csv_rows(output_path, rows, fieldnames=fieldnames)


def create_train_test_split(
    data_dir: str | Path,
    train_out: str | Path,
    test_out: str | Path,
    test_ratio: float,
    seed: int,
) -> tuple[Path, Path]:
    """Create deterministic train/test metadata splits from the generated dataset."""
    rows = load_metadata_rows(data_dir)
    labels = [row["label"] for row in rows]
    train_rows, test_rows = train_test_split(
        rows,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )
    train_path = write_split_metadata(list(train_rows), train_out)
    test_path = write_split_metadata(list(test_rows), test_out)
    return train_path, test_path


def main() -> None:
    """Validate the generated dataset and print a readable summary."""
    args = parse_args()
    report = validate_generated_dataset(args.data_dir, min_images_per_class=args.min_images_per_class)

    print(f"Dataset directory: {report['data_dir']}")
    print(f"Checked images: {report['checked_images']}")
    print("Class counts:")
    for class_name in CLASS_NAMES:
        print(f"  - {class_name}: {report['class_counts'].get(class_name, 0)}")

    if report["errors"]:
        print("Validation failed:")
        for error in report["errors"]:
            print(f"  - {error}")
        raise SystemExit(1)

    print("Validation passed.")

    if args.train_out and args.test_out:
        train_path, test_path = create_train_test_split(
            data_dir=args.data_dir,
            train_out=args.train_out,
            test_out=args.test_out,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print(f"Train split written to: {train_path}")
        print(f"Test split written to: {test_path}")


if __name__ == "__main__":
    main()
