"""Local scoring interface for image-model inference."""

from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_utils import CLASS_NAMES, PRODUCT_TYPES, load_json
from src.training_utils import (
    build_image_transform,
    load_baseline_bundle,
    load_grayscale_feature_vector,
    load_torch_checkpoint,
    product_type_one_hot,
)


_RUNTIME: "ScoringRuntime | None" = None


@dataclass
class ScoringRuntime:
    """Hold a loaded local model and the information needed to score one image."""

    model_name: str
    artifact_path: Path
    class_names: list[str]
    bundle: dict[str, Any] | None = None
    torch_model: torch.nn.Module | None = None
    image_size: tuple[int, int] | None = None
    device: torch.device | None = None


def error_response(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a consistent error payload."""
    return {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }


def success_response(
    runtime: ScoringRuntime,
    request_item: dict[str, Any],
    label: str,
    scores: dict[str, float],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a consistent success payload."""
    confidence = max(scores.values()) if scores else 0.0
    priority_label = "stain"
    return {
        "status": "ok",
        "model": {
            "name": runtime.model_name,
            "artifact": str(runtime.artifact_path),
            "class_names": runtime.class_names,
        },
        "input": {
            "image_path": request_item.get("image_path"),
            "product_type": request_item.get("product_type"),
        },
        "prediction": {
            "label": label,
            "confidence": round(confidence, 6),
            "scores": {class_name: round(float(scores[class_name]), 6) for class_name in runtime.class_names},
            "priority_class": priority_label,
            "priority_class_score": round(float(scores.get(priority_label, 0.0)), 6),
        },
        "warnings": warnings or [],
    }


def choose_model_artifact(model_dir: str | Path, model_name: str = "auto") -> tuple[str, Path]:
    """Choose the requested or recommended local model artifact."""
    model_path = Path(model_dir)
    baseline_path = model_path / "baseline_model.joblib"
    main_path = model_path / "main_model.pt"
    selected_path = model_path / "selected_model.json"

    if model_name == "baseline":
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline model artifact not found: {baseline_path}")
        return "baseline", baseline_path

    if model_name == "main":
        if not main_path.exists():
            raise FileNotFoundError(f"Main model artifact not found: {main_path}")
        return "main", main_path

    if selected_path.exists():
        selection = load_json(selected_path)
        recommended = selection.get("recommended_model")
        if recommended == "main" and main_path.exists():
            return "main", main_path
        if recommended == "baseline" and baseline_path.exists():
            return "baseline", baseline_path

    if main_path.exists():
        return "main", main_path
    if baseline_path.exists():
        return "baseline", baseline_path

    raise FileNotFoundError(
        f"No supported model artifacts were found in {model_path}. "
        "Expected baseline_model.joblib and/or main_model.pt."
    )


def load_runtime(model_dir: str | Path = "models", model_name: str = "auto") -> ScoringRuntime:
    """Load the selected local model artifact into memory."""
    chosen_model_name, artifact_path = choose_model_artifact(model_dir, model_name=model_name)

    if chosen_model_name == "baseline":
        bundle = load_baseline_bundle(artifact_path)
        return ScoringRuntime(
            model_name=chosen_model_name,
            artifact_path=artifact_path,
            class_names=list(bundle["class_names"]),
            bundle=bundle,
            image_size=tuple(bundle["image_size"]),
        )

    device = torch.device("cpu")
    model, checkpoint = load_torch_checkpoint(artifact_path, device=device)
    return ScoringRuntime(
        model_name=chosen_model_name,
        artifact_path=artifact_path,
        class_names=list(checkpoint["class_names"]),
        torch_model=model,
        image_size=tuple(checkpoint.get("image_size", [160, 160])),
        device=device,
    )


def init(model_dir: str | Path = "models", model_name: str = "auto") -> None:
    """Initialize the global local scoring runtime."""
    global _RUNTIME
    _RUNTIME = load_runtime(model_dir=model_dir, model_name=model_name)


def ensure_runtime(model_dir: str | Path = "models", model_name: str = "auto") -> ScoringRuntime:
    """Load the runtime lazily if it has not been initialized yet."""
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = load_runtime(model_dir=model_dir, model_name=model_name)
    return _RUNTIME


def normalize_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize direct or batched payloads into a single-image request item."""
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    if "inputs" in payload:
        inputs = payload["inputs"]
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("Payload field 'inputs' must be a non-empty list.")
        if len(inputs) != 1:
            raise ValueError("This local scorer currently supports exactly one input item per request.")
        item = inputs[0]
        if not isinstance(item, dict):
            raise ValueError("Each entry in 'inputs' must be an object.")
        return item

    return payload


def open_request_image(item: dict[str, Any]) -> Image.Image:
    """Open an image from a request item using either a path or base64 content."""
    image_path = item.get("image_path")
    image_base64 = item.get("image_base64")

    if bool(image_path) == bool(image_base64):
        raise ValueError("Provide exactly one of 'image_path' or 'image_base64'.")

    try:
        if image_path:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            with Image.open(path) as image:
                return image.convert("RGB")

        image_bytes = base64.b64decode(image_base64)
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except FileNotFoundError:
        raise
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or unreadable image input: {exc}") from exc
    except base64.binascii.Error as exc:  # pragma: no cover - depends on bad request content
        raise ValueError(f"Invalid base64 image content: {exc}") from exc


def validate_product_type(product_type: str | None) -> list[str]:
    """Validate product type and return any non-fatal warnings."""
    warnings: list[str] = []
    if product_type is None:
        warnings.append(
            "No product_type was provided. Baseline inference will fall back to an all-zero metadata vector."
        )
        return warnings

    if product_type not in PRODUCT_TYPES:
        valid = ", ".join(PRODUCT_TYPES)
        raise ValueError(f"Invalid product_type '{product_type}'. Expected one of: {valid}.")

    return warnings


def baseline_scores(runtime: ScoringRuntime, image: Image.Image, product_type: str | None) -> dict[str, float]:
    """Score one image using the saved scikit-learn baseline model."""
    if runtime.bundle is None or runtime.image_size is None:
        raise RuntimeError("Baseline runtime is not initialized correctly.")

    grayscale = image.convert("L").resize(runtime.image_size)
    pixel_array = np.asarray(grayscale, dtype=np.float32).flatten() / 255.0
    metadata_array = product_type_one_hot(product_type or "")
    features = np.concatenate([pixel_array, metadata_array], axis=0).reshape(1, -1)
    probabilities = runtime.bundle["pipeline"].predict_proba(features)[0]
    return {
        class_name: float(probability)
        for class_name, probability in zip(runtime.bundle["pipeline"].classes_.tolist(), probabilities.tolist())
    }


def main_model_scores(runtime: ScoringRuntime, image: Image.Image) -> dict[str, float]:
    """Score one image using the saved torch transfer-learning model."""
    if runtime.torch_model is None or runtime.image_size is None or runtime.device is None:
        raise RuntimeError("Main-model runtime is not initialized correctly.")

    transform = build_image_transform(train=False, image_size=runtime.image_size)
    tensor = transform(image).unsqueeze(0).to(runtime.device)

    runtime.torch_model.eval()
    with torch.no_grad():
        logits = runtime.torch_model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {
        class_name: float(probability)
        for class_name, probability in zip(runtime.class_names, probabilities.tolist())
    }


def score_item(runtime: ScoringRuntime, item: dict[str, Any]) -> dict[str, Any]:
    """Score a single normalized request item."""
    product_type = item.get("product_type")
    warnings = validate_product_type(product_type)
    image = open_request_image(item)

    if runtime.model_name == "baseline":
        scores = baseline_scores(runtime, image=image, product_type=product_type)
    else:
        scores = main_model_scores(runtime, image=image)

    label = max(scores, key=scores.get)
    return success_response(runtime, item, label=label, scores=scores, warnings=warnings)


def run(payload: dict[str, Any], model_dir: str | Path = "models", model_name: str = "auto") -> dict[str, Any]:
    """Score one local inference payload and return a structured response."""
    try:
        runtime = ensure_runtime(model_dir=model_dir, model_name=model_name)
        item = normalize_request(payload)
        return score_item(runtime, item)
    except FileNotFoundError as exc:
        return error_response("file_not_found", str(exc))
    except ValueError as exc:
        return error_response("invalid_input", str(exc))
    except RuntimeError as exc:
        return error_response("runtime_error", str(exc))
    except Exception as exc:  # pragma: no cover - safety net for unexpected runtime failures
        return error_response("unexpected_error", str(exc))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local scoring."""
    parser = argparse.ArgumentParser(description="Score a single local image request.")
    parser.add_argument("--request_json", help="Path to a JSON request file.")
    parser.add_argument("--model_dir", default="models", help="Directory containing local model artifacts.")
    parser.add_argument(
        "--model_name",
        default="auto",
        choices=["auto", "baseline", "main"],
        help="Which local model artifact to load.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for scoring a request JSON file."""
    args = parse_args()
    if not args.request_json:
        raise SystemExit("Please provide --request_json <path-to-request.json>.")

    payload = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    response = run(payload, model_dir=args.model_dir, model_name=args.model_name)
    print(json.dumps(response, indent=2))
    if response.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
