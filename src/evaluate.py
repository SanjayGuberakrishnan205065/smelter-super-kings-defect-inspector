"""Evaluate saved image models and produce comparison artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_utils import DEFAULT_IMAGE_SIZE, ensure_directory, save_json
from src.training_utils import (
    evaluate_baseline_bundle,
    evaluate_torch_model,
    load_baseline_bundle,
    load_metadata_rows_from_file,
    load_torch_checkpoint,
    select_preferred_model,
)


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments for saved model artifacts."""
    parser = argparse.ArgumentParser(description="Evaluate baseline and main image models.")
    parser.add_argument("--data_dir", default="data/images", help="Dataset root containing image files.")
    parser.add_argument("--test_metadata", default="split/test_metadata.csv", help="Test metadata CSV.")
    parser.add_argument("--model_dir", default="models", help="Directory containing saved model artifacts.")
    parser.add_argument("--metrics_dir", default="metrics", help="Directory where metrics are written.")
    parser.add_argument(
        "--model_name",
        default="all",
        choices=["all", "baseline", "main"],
        help="Which saved model artifact to evaluate.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for torch evaluation.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--min_macro_f1", type=float, default=0.60, help="Macro F1 quality gate.")
    parser.add_argument("--min_stain_recall", type=float, default=0.50, help="Stain recall guardrail.")
    parser.add_argument("--min_stain_f1", type=float, default=0.50, help="Stain F1 guardrail.")
    parser.add_argument("--image_width", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Evaluation image width.")
    parser.add_argument("--image_height", type=int, default=DEFAULT_IMAGE_SIZE[1], help="Evaluation image height.")
    return parser.parse_args()


def main() -> None:
    """Evaluate saved models and emit JSON artifacts."""
    args = parse_args()
    metrics_dir = ensure_directory(args.metrics_dir)
    rows = load_metadata_rows_from_file(args.test_metadata)

    baseline_metrics = None
    main_metrics = None

    if args.model_name in {"all", "baseline"}:
        baseline_bundle = load_baseline_bundle(Path(args.model_dir) / "baseline_model.joblib")
        baseline_metrics = evaluate_baseline_bundle(baseline_bundle, rows=rows, data_dir=args.data_dir)
        save_json(metrics_dir / "baseline_metrics.json", baseline_metrics)
        print(f"Saved baseline metrics to: {metrics_dir / 'baseline_metrics.json'}")

    if args.model_name in {"all", "main"}:
        device = torch.device("cpu")
        model, checkpoint = load_torch_checkpoint(Path(args.model_dir) / "main_model.pt", device=device)
        image_size = tuple(checkpoint.get("image_size", [args.image_width, args.image_height]))
        main_metrics = evaluate_torch_model(
            model=model,
            rows=rows,
            data_dir=args.data_dir,
            device=device,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        save_json(metrics_dir / "main_metrics.json", main_metrics)
        print(f"Saved main metrics to: {metrics_dir / 'main_metrics.json'}")

    if baseline_metrics and main_metrics:
        comparison = select_preferred_model(
            baseline_metrics=baseline_metrics,
            main_metrics=main_metrics,
            min_macro_f1=args.min_macro_f1,
            min_stain_recall=args.min_stain_recall,
            min_stain_f1=args.min_stain_f1,
        )
        save_json(metrics_dir / "comparison.json", comparison)
        save_json(Path(args.model_dir) / "selected_model.json", comparison)
        print(f"Saved comparison artifact to: {metrics_dir / 'comparison.json'}")
        print(f"Saved selected-model summary to: {Path(args.model_dir) / 'selected_model.json'}")


if __name__ == "__main__":
    main()
