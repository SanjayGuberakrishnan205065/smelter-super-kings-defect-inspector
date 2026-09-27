"""Train the classical image baseline for the capstone."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_utils import DEFAULT_BASELINE_IMAGE_SIZE, DEFAULT_TRAIN_TEST_SEED, ensure_directory, save_json
from src.training_utils import (
    build_baseline_feature_matrix,
    build_baseline_pipeline,
    load_metadata_rows_from_file,
)


def parse_args() -> argparse.Namespace:
    """Parse baseline training arguments."""
    parser = argparse.ArgumentParser(description="Train the classical baseline image classifier.")
    parser.add_argument("--data_dir", default="data/images", help="Dataset root containing class folders.")
    parser.add_argument("--train_metadata", default="split/train_metadata.csv", help="Training metadata CSV.")
    parser.add_argument("--model_dir", default="models", help="Directory where model artifacts are saved.")
    parser.add_argument("--image_width", type=int, default=DEFAULT_BASELINE_IMAGE_SIZE[0], help="Baseline resize width.")
    parser.add_argument("--image_height", type=int, default=DEFAULT_BASELINE_IMAGE_SIZE[1], help="Baseline resize height.")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_TEST_SEED, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    """Train and save the scikit-learn baseline model bundle."""
    args = parse_args()
    rows = load_metadata_rows_from_file(args.train_metadata)
    image_size = (args.image_width, args.image_height)
    features, labels = build_baseline_feature_matrix(rows, data_dir=args.data_dir, image_size=image_size)

    pipeline = build_baseline_pipeline()
    classifier = pipeline.named_steps["classifier"]
    classifier.random_state = args.seed
    pipeline.fit(features, labels)

    model_dir = ensure_directory(args.model_dir)
    model_path = model_dir / "baseline_model.joblib"
    bundle = {
        "pipeline": pipeline,
        "class_names": list(pipeline.classes_),
        "image_size": list(image_size),
        "feature_type": "resized_grayscale_pixels_plus_product_type",
        "data_marking": "SYNTHETIC",
    }
    joblib.dump(bundle, model_path)

    summary_path = model_dir / "baseline_model_summary.json"
    save_json(
        summary_path,
        {
            "artifact": str(model_path),
            "class_names": bundle["class_names"],
            "image_size": bundle["image_size"],
            "feature_type": bundle["feature_type"],
            "data_marking": bundle["data_marking"],
            "train_rows": len(rows),
        },
    )

    print(f"Saved baseline model to: {model_path}")
    print(f"Saved baseline summary to: {summary_path}")


if __name__ == "__main__":
    main()
