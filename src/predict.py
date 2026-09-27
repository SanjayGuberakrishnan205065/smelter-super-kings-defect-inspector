"""CLI helper for single-image local prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.score import init, run


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-image prediction."""
    parser = argparse.ArgumentParser(description="Run a single local image prediction.")
    parser.add_argument("--image_path", required=True, help="Path to the image file to score.")
    parser.add_argument("--product_type", help="Optional product type: coil, sheet, plate, or strip.")
    parser.add_argument("--model_dir", default="models", help="Directory containing local model artifacts.")
    parser.add_argument(
        "--model_name",
        default="auto",
        choices=["auto", "baseline", "main"],
        help="Which local model artifact to load.",
    )
    parser.add_argument("--response_out", help="Optional path to save the JSON response.")
    return parser.parse_args()


def main() -> None:
    """Run a single local prediction and print the JSON response."""
    args = parse_args()
    init(model_dir=args.model_dir, model_name=args.model_name)

    payload = {
        "inputs": [
            {
                "image_path": args.image_path,
                "product_type": args.product_type,
            }
        ]
    }
    response = run(payload, model_dir=args.model_dir, model_name=args.model_name)
    response_text = json.dumps(response, indent=2)
    print(response_text)

    if args.response_out:
        output_path = Path(args.response_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response_text, encoding="utf-8")

    if response.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
