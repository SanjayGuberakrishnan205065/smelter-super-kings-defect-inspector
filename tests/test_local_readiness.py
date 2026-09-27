"""Local-capstone readiness checks for generated artifacts and inference."""

from __future__ import annotations

import json
from pathlib import Path

from src.model_utils import CLASS_NAMES, PRIORITY_CLASS
from src.prep import load_metadata_rows
from src.score import run


MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
SPLIT_DIR = Path("split")
DATA_DIR = Path("data/images")


def read_json(path: str | Path) -> dict:
    """Read a JSON file from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_split_artifacts_exist() -> None:
    """Verify the train/test metadata split artifacts exist."""
    assert (SPLIT_DIR / "train_metadata.csv").exists()
    assert (SPLIT_DIR / "test_metadata.csv").exists()


def test_model_artifacts_exist() -> None:
    """Verify the saved local model artifacts exist."""
    assert (MODELS_DIR / "baseline_model.joblib").exists()
    assert (MODELS_DIR / "baseline_model_summary.json").exists()
    assert (MODELS_DIR / "main_model.pt").exists()
    assert (MODELS_DIR / "selected_model.json").exists()


def test_metrics_artifacts_exist() -> None:
    """Verify the expected local evaluation artifacts exist."""
    assert (METRICS_DIR / "baseline_metrics.json").exists()
    assert (METRICS_DIR / "main_metrics.json").exists()
    assert (METRICS_DIR / "comparison.json").exists()
    assert (METRICS_DIR / "main_training_history.json").exists()


def test_metrics_include_priority_class_reporting() -> None:
    """Verify saved metrics report macro F1 and stain-specific results."""
    for metrics_name in ["baseline_metrics.json", "main_metrics.json"]:
        payload = read_json(METRICS_DIR / metrics_name)
        assert payload["class_names"] == CLASS_NAMES
        assert "macro_f1" in payload
        assert "priority_class" in payload
        assert payload["priority_class"]["label"] == PRIORITY_CLASS
        assert "recall" in payload["priority_class"]
        assert "f1" in payload["priority_class"]
        assert set(payload["per_class"].keys()) == set(CLASS_NAMES)


def test_selected_model_summary_is_valid() -> None:
    """Verify the selected-model summary captures the recommendation cleanly."""
    payload = read_json(MODELS_DIR / "selected_model.json")
    assert payload["recommended_model"] in {"baseline", "main"}
    assert payload["quality_gate"]["priority_class"] == PRIORITY_CLASS


def test_local_scoring_succeeds_for_one_generated_sample() -> None:
    """Verify local scoring returns a full prediction payload for one generated image."""
    row = load_metadata_rows(DATA_DIR)[0]
    response = run(
        {
            "inputs": [
                {
                    "image_path": str(DATA_DIR / row["image_path"]),
                    "product_type": row["product_type"],
                }
            ]
        },
        model_dir=MODELS_DIR,
        model_name="auto",
    )

    assert response["status"] == "ok", response
    assert response["prediction"]["label"] in CLASS_NAMES
    assert set(response["prediction"]["scores"].keys()) == set(CLASS_NAMES)
    assert 0.0 <= response["prediction"]["confidence"] <= 1.0


def test_local_scoring_returns_structured_error_for_missing_file() -> None:
    """Verify invalid local inference input returns a helpful structured error."""
    response = run(
        {
            "inputs": [
                {
                    "image_path": "data/images/missing_file.png",
                    "product_type": "coil",
                }
            ]
        },
        model_dir=MODELS_DIR,
        model_name="auto",
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "file_not_found"
