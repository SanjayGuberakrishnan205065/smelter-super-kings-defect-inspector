"""Dataset-structure tests for the synthetic image-track capstone data."""

from __future__ import annotations

from pathlib import Path

from src.model_utils import (
    CLASS_NAMES,
    DATASET_NOTICE_FILENAME,
    DEFAULT_IMAGE_SIZE,
    METADATA_FILENAME,
    PRIORITY_CLASS,
)
from src.prep import load_metadata_rows, validate_generated_dataset


DATA_DIR = Path("data/images")
MIN_IMAGES_PER_CLASS = 20


def test_expected_project_directories_exist() -> None:
    """Verify the core project directories are present."""
    for relative_path in ["app", "src", "tests", "specs", "data/images", "models", "metrics"]:
        assert Path(relative_path).exists(), f"Missing expected path: {relative_path}"


def test_problem_documents_exist() -> None:
    """Verify the core problem-definition documents are present."""
    assert Path("specs/problem_brief.md").exists()
    assert Path("specs/capstone_spec.md").exists()
    assert Path("specs/validation_report.md").exists()


def test_generated_dataset_exists() -> None:
    """Verify that generated dataset artifacts exist before downstream steps run."""
    assert DATA_DIR.exists(), "Expected dataset directory at data/images."
    assert (DATA_DIR / DATASET_NOTICE_FILENAME).exists(), "Missing synthetic dataset notice."
    assert (DATA_DIR / METADATA_FILENAME).exists(), "Missing dataset metadata.csv."


def test_generated_dataset_is_structurally_valid() -> None:
    """Validate the generated dataset folders, metadata, and image files."""
    report = validate_generated_dataset(
        DATA_DIR,
        min_images_per_class=MIN_IMAGES_PER_CLASS,
        expected_size=DEFAULT_IMAGE_SIZE,
    )
    assert not report["errors"], "Dataset validation errors:\n- " + "\n- ".join(report["errors"])


def test_all_expected_classes_have_coverage() -> None:
    """Verify every target class is present with enough examples for local review."""
    rows = load_metadata_rows(DATA_DIR)
    counts = {class_name: 0 for class_name in CLASS_NAMES}

    for row in rows:
        counts[row["label"]] += 1

    for class_name in CLASS_NAMES:
        assert counts[class_name] >= MIN_IMAGES_PER_CLASS, (
            f"Class '{class_name}' has {counts[class_name]} images; "
            f"expected at least {MIN_IMAGES_PER_CLASS}."
        )


def test_dataset_is_clearly_marked_synthetic() -> None:
    """Verify the dataset-level and row-level markers identify the data as synthetic."""
    notice_text = (DATA_DIR / DATASET_NOTICE_FILENAME).read_text(encoding="utf-8")
    assert "SYNTHETIC" in notice_text.upper()

    rows = load_metadata_rows(DATA_DIR)
    assert rows, "Expected metadata rows after dataset generation."
    for row in rows:
        assert row["is_synthetic"].lower() == "true"
        assert "synthetic" in row["synthetic_note"].lower()


def test_priority_class_has_explicit_dataset_coverage() -> None:
    """Verify the priority stain class is present in the generated dataset."""
    rows = load_metadata_rows(DATA_DIR)
    priority_rows = [row for row in rows if row["label"] == PRIORITY_CLASS]
    assert len(priority_rows) >= MIN_IMAGES_PER_CLASS


def test_dataset_metadata_paths_resolve_to_real_files() -> None:
    """Verify every metadata image reference points to an existing class file."""
    rows = load_metadata_rows(DATA_DIR)
    for row in rows:
        assert (DATA_DIR / row["image_path"]).exists(), f"Missing image for metadata row: {row}"
