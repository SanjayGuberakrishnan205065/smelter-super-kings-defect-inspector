"""Build a clean local inference request for later reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_utils import CLASS_NAMES, PRODUCT_TYPES, get_metadata_path, read_csv_rows


def choose_sample_image(
    data_dir: str | Path,
    label: str | None = None,
    sample_index: int = 0,
) -> tuple[Path, str]:
    """Choose a sample image path from metadata or class folders."""
    data_path = Path(data_dir)
    metadata_path = get_metadata_path(data_path)

    if metadata_path.exists():
        rows = read_csv_rows(metadata_path)
        if label:
            rows = [row for row in rows if row["label"] == label]
        if not rows:
            raise FileNotFoundError("No matching sample rows were found in metadata.csv.")
        row = rows[sample_index % len(rows)]
        return data_path / row["image_path"], row.get("product_type", "")

    search_label = label or CLASS_NAMES[0]
    class_dir = data_path / search_label
    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")
    images = sorted(class_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found under {class_dir}")
    return images[sample_index % len(images)], ""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for building a request JSON file."""
    parser = argparse.ArgumentParser(description="Create a JSON request for local image scoring.")
    parser.add_argument("--image_path", help="Explicit image path to include in the request.")
    parser.add_argument("--data_dir", default="data/images", help="Dataset root used to choose a sample image.")
    parser.add_argument(
        "--sample_label",
        choices=CLASS_NAMES,
        help="Optional class label to choose a sample image from when --image_path is omitted.",
    )
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index within the chosen class.")
    parser.add_argument(
        "--product_type",
        choices=PRODUCT_TYPES,
        help="Optional product type to include in the request. If omitted, metadata is used when available.",
    )
    parser.add_argument("--out", help="Optional output JSON path. If omitted, the request is printed only.")
    return parser.parse_args()


def main() -> None:
    """Create a JSON request for a single local image prediction."""
    args = parse_args()

    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.exists():
            raise SystemExit(f"Image file not found: {image_path}")
        inferred_product_type = ""
    else:
        image_path, inferred_product_type = choose_sample_image(
            data_dir=args.data_dir,
            label=args.sample_label,
            sample_index=args.sample_index,
        )

    request_payload = {
        "inputs": [
            {
                "image_path": str(image_path),
                "product_type": args.product_type or inferred_product_type or None,
            }
        ]
    }

    payload_text = json.dumps(request_payload, indent=2)
    print(payload_text)

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload_text, encoding="utf-8")
        print(f"Saved request to: {output_path}")


if __name__ == "__main__":
    main()
