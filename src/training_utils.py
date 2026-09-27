"""Reusable training, evaluation, and artifact helpers for image models."""

from __future__ import annotations

from collections import Counter
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from src.model_utils import (
    CLASS_NAMES,
    DEFAULT_BASELINE_IMAGE_SIZE,
    DEFAULT_IMAGE_SIZE,
    PRIORITY_CLASS,
    PRODUCT_TYPES,
    read_csv_rows,
)

try:
    from torchvision.models import MobileNet_V3_Small_Weights
except ImportError:  # pragma: no cover - depends on torchvision version
    MobileNet_V3_Small_Weights = None


def load_metadata_rows_from_file(path: str | Path) -> list[dict[str, str]]:
    """Load metadata rows from a CSV file path."""
    return read_csv_rows(path)


def split_rows_by_label(
    rows: list[dict[str, str]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a deterministic stratified split from metadata rows."""
    labels = [row["label"] for row in rows]
    train_rows, test_rows = train_test_split(
        rows,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )
    return list(train_rows), list(test_rows)


def resolve_image_path(data_dir: str | Path, row: dict[str, str]) -> Path:
    """Resolve an image path from a metadata row."""
    return Path(data_dir) / row["image_path"]


def product_type_one_hot(product_type: str) -> np.ndarray:
    """Encode a product type as a fixed one-hot vector."""
    vector = np.zeros(len(PRODUCT_TYPES), dtype=np.float32)
    if product_type in PRODUCT_TYPES:
        vector[PRODUCT_TYPES.index(product_type)] = 1.0
    return vector


def load_grayscale_feature_vector(
    image_path: str | Path,
    image_size: tuple[int, int] = DEFAULT_BASELINE_IMAGE_SIZE,
) -> np.ndarray:
    """Load an image as a resized grayscale feature vector."""
    with Image.open(image_path) as image:
        grayscale = image.convert("L").resize(image_size)
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
    return array.flatten()


def build_baseline_feature_matrix(
    rows: list[dict[str, str]],
    data_dir: str | Path,
    image_size: tuple[int, int] = DEFAULT_BASELINE_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert metadata rows into classical-ML feature and label arrays."""
    features: list[np.ndarray] = []
    labels: list[str] = []

    for row in rows:
        image_features = load_grayscale_feature_vector(resolve_image_path(data_dir, row), image_size=image_size)
        metadata_features = product_type_one_hot(row.get("product_type", ""))
        features.append(np.concatenate([image_features, metadata_features], axis=0))
        labels.append(row["label"])

    return np.stack(features), np.asarray(labels)


def build_baseline_pipeline() -> Pipeline:
    """Build the baseline scikit-learn image classifier."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def get_label_counts(rows: list[dict[str, str]]) -> Counter[str]:
    """Count examples per class label in a metadata row collection."""
    return Counter(row["label"] for row in rows)


def build_class_weight_tensor(
    rows: list[dict[str, str]],
    priority_label: str = PRIORITY_CLASS,
    priority_boost: float = 2.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a class-weight tensor with an explicit boost for the priority class."""
    label_counts = get_label_counts(rows)
    total_count = sum(label_counts.values())
    num_classes = len(CLASS_NAMES)

    weights = []
    for class_name in CLASS_NAMES:
        class_count = max(label_counts.get(class_name, 1), 1)
        base_weight = total_count / (num_classes * class_count)
        if class_name == priority_label:
            base_weight *= priority_boost
        weights.append(base_weight)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weight_tensor = weight_tensor.to(device)
    return weight_tensor


def build_weighted_sampler(
    rows: list[dict[str, str]],
    priority_label: str = PRIORITY_CLASS,
    priority_boost: float = 1.5,
) -> WeightedRandomSampler:
    """Build a weighted sampler that oversamples the priority class modestly."""
    label_counts = get_label_counts(rows)
    sample_weights = []

    for row in rows:
        label = row["label"]
        class_count = max(label_counts.get(label, 1), 1)
        weight = 1.0 / class_count
        if label == priority_label:
            weight *= priority_boost
        sample_weights.append(weight)

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(rows),
        replacement=True,
    )


def compute_classification_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str] | None = None,
    priority_label: str = PRIORITY_CLASS,
) -> dict[str, Any]:
    """Compute overall, per-class, and priority-class metrics."""
    ordered_labels = labels or list(CLASS_NAMES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=ordered_labels,
        zero_division=0,
    )
    macro_f1 = float(np.mean(f1))
    accuracy = float(accuracy_score(y_true, y_pred))
    confusion = confusion_matrix(y_true, y_pred, labels=ordered_labels)

    per_class = {}
    for index, label in enumerate(ordered_labels):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    priority_metrics = {
        "label": priority_label,
        "recall": per_class[priority_label]["recall"],
        "f1": per_class[priority_label]["f1"],
        "support": per_class[priority_label]["support"],
    }

    return {
        "class_names": ordered_labels,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "priority_class": priority_metrics,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": ordered_labels,
            "values": confusion.tolist(),
        },
    }


def select_preferred_model(
    baseline_metrics: dict[str, Any],
    main_metrics: dict[str, Any],
    min_macro_f1: float,
    min_stain_recall: float = 0.0,
    min_stain_f1: float = 0.0,
) -> dict[str, Any]:
    """Compare model metrics and return a deployment recommendation."""

    def score_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
        priority = metrics["priority_class"]
        return (
            float(metrics["macro_f1"]),
            float(priority["f1"]),
            float(priority["recall"]),
            float(metrics["accuracy"]),
        )

    baseline_score = score_tuple(baseline_metrics)
    main_score = score_tuple(main_metrics)
    recommended = "main" if main_score >= baseline_score else "baseline"

    return {
        "quality_gate": {
            "metric": "macro_f1",
            "threshold": float(min_macro_f1),
            "baseline_pass": baseline_metrics["macro_f1"] >= min_macro_f1,
            "main_pass": main_metrics["macro_f1"] >= min_macro_f1,
            "priority_class": PRIORITY_CLASS,
            "min_stain_recall": float(min_stain_recall),
            "min_stain_f1": float(min_stain_f1),
            "baseline_stain_recall_pass": baseline_metrics["priority_class"]["recall"] >= min_stain_recall,
            "baseline_stain_f1_pass": baseline_metrics["priority_class"]["f1"] >= min_stain_f1,
            "main_stain_recall_pass": main_metrics["priority_class"]["recall"] >= min_stain_recall,
            "main_stain_f1_pass": main_metrics["priority_class"]["f1"] >= min_stain_f1,
        },
        "baseline": {
            "macro_f1": baseline_metrics["macro_f1"],
            "stain_f1": baseline_metrics["priority_class"]["f1"],
            "stain_recall": baseline_metrics["priority_class"]["recall"],
            "accuracy": baseline_metrics["accuracy"],
        },
        "main": {
            "macro_f1": main_metrics["macro_f1"],
            "stain_f1": main_metrics["priority_class"]["f1"],
            "stain_recall": main_metrics["priority_class"]["recall"],
            "accuracy": main_metrics["accuracy"],
        },
        "recommended_model": recommended,
        "rationale": [
            "Primary ranking uses macro F1.",
            "Tie-breaks favor stain F1, then stain recall, then accuracy.",
            "The priority defect remains stain throughout evaluation.",
        ],
    }


def compute_priority_selection_score(
    metrics: dict[str, Any],
    macro_weight: float = 1.0,
    stain_f1_weight: float = 0.75,
    stain_recall_weight: float = 0.5,
) -> float:
    """Compute a validation score that preserves macro F1 while rewarding stain performance."""
    priority_metrics = metrics["priority_class"]
    return (
        float(metrics["macro_f1"]) * macro_weight
        + float(priority_metrics["f1"]) * stain_f1_weight
        + float(priority_metrics["recall"]) * stain_recall_weight
    )


class MetadataImageDataset(Dataset):
    """Torch dataset backed by the generated metadata CSV and image files."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        data_dir: str | Path,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.rows = rows
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform or build_image_transform(train=False, image_size=image_size)
        self.label_to_index = {label: index for index, label in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = resolve_image_path(self.data_dir, row)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, self.label_to_index[row["label"]]


def build_image_transform(train: bool, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Build the transform stack for image training or evaluation."""
    transform_steps: list[Any] = [transforms.Resize(image_size)]
    if train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04, hue=0.02),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ]
    )
    return transforms.Compose(transform_steps)


def build_transfer_model(
    num_classes: int,
    pretrained_mode: str = "auto",
    freeze_features: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a CPU-friendly MobileNet transfer-learning model."""
    transfer_info: dict[str, Any] = {
        "architecture": "mobilenet_v3_small",
        "requested_pretrained_mode": pretrained_mode,
        "used_pretrained_weights": False,
        "resolved_pretrained_mode": "none",
    }

    weights = None
    if pretrained_mode in {"auto", "default"} and MobileNet_V3_Small_Weights is not None:
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            transfer_info["used_pretrained_weights"] = True
            transfer_info["resolved_pretrained_mode"] = "imagenet_default"
        except Exception as exc:  # pragma: no cover - enum access is stable, download is not
            warnings.warn(f"Unable to access pretrained weights: {exc}. Falling back to random initialization.")
            weights = None

    try:
        model = models.mobilenet_v3_small(weights=weights)
    except Exception as exc:  # pragma: no cover - depends on local cache/network state
        warnings.warn(
            f"Unable to load pretrained MobileNet weights ({exc}). Falling back to random initialization."
        )
        model = models.mobilenet_v3_small(weights=None)
        transfer_info["used_pretrained_weights"] = False
        transfer_info["resolved_pretrained_mode"] = "none"
        transfer_info["fallback_reason"] = str(exc)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        transfer_info["feature_extractor_frozen"] = True
    else:
        transfer_info["feature_extractor_frozen"] = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model, transfer_info


def create_data_loader(
    rows: list[dict[str, str]],
    data_dir: str | Path,
    batch_size: int,
    image_size: tuple[int, int],
    train: bool,
    num_workers: int = 0,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    """Create a DataLoader for metadata-backed images."""
    dataset = MetadataImageDataset(
        rows=rows,
        data_dir=data_dir,
        image_size=image_size,
        transform=build_image_transform(train=train, image_size=image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> float:
    """Train the model for one epoch and return mean loss."""
    model.train()
    criterion = criterion or nn.CrossEntropyLoss()
    running_loss = 0.0
    sample_count = 0

    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


def evaluate_torch_model(
    model: nn.Module,
    rows: list[dict[str, str]],
    data_dir: str | Path,
    device: torch.device,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Run torch inference over metadata rows and compute metrics."""
    data_loader = create_data_loader(
        rows=rows,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        train=False,
        num_workers=num_workers,
    )
    index_to_label = {index: label for index, label in enumerate(CLASS_NAMES)}
    y_true: list[str] = []
    y_pred: list[str] = []

    model.eval()
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            logits = model(features)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            y_true.extend(index_to_label[index] for index in labels.tolist())
            y_pred.extend(index_to_label[index] for index in predictions)

    return compute_classification_metrics(y_true, y_pred)


def save_torch_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save a torch checkpoint to disk."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_torch_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved torch checkpoint and reconstruct the model."""
    checkpoint = torch.load(Path(path), map_location=device)
    model, transfer_info = build_transfer_model(
        num_classes=len(checkpoint["class_names"]),
        pretrained_mode="none",
        freeze_features=bool(checkpoint.get("transfer_info", {}).get("feature_extractor_frozen", True)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    transfer_info.update(checkpoint.get("transfer_info", {}))
    return model, checkpoint


def evaluate_baseline_bundle(
    bundle: dict[str, Any],
    rows: list[dict[str, str]],
    data_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate a saved baseline bundle on metadata rows."""
    features, labels = build_baseline_feature_matrix(
        rows=rows,
        data_dir=data_dir,
        image_size=tuple(bundle["image_size"]),
    )
    predictions = bundle["pipeline"].predict(features)
    return compute_classification_metrics(labels.tolist(), predictions.tolist())


def load_baseline_bundle(path: str | Path) -> dict[str, Any]:
    """Load a saved baseline model bundle."""
    return joblib.load(Path(path))
