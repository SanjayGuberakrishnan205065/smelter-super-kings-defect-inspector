"""Train the main CPU-friendly transfer-learning image classifier."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_utils import (
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_TEST_SEED,
    ensure_directory,
    save_json,
)
from src.training_utils import (
    build_transfer_model,
    build_class_weight_tensor,
    build_weighted_sampler,
    compute_priority_selection_score,
    create_data_loader,
    evaluate_torch_model,
    load_metadata_rows_from_file,
    save_torch_checkpoint,
    split_rows_by_label,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    """Parse training arguments for the main image model."""
    parser = argparse.ArgumentParser(description="Train the main transfer-learning image classifier.")
    parser.add_argument("--data_dir", default="data/images", help="Dataset root containing class folders.")
    parser.add_argument("--train_metadata", default="split/train_metadata.csv", help="Training metadata CSV.")
    parser.add_argument("--model_dir", default="models", help="Directory where model artifacts are saved.")
    parser.add_argument("--metrics_dir", default="metrics", help="Directory where training artifacts are saved.")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_TEST_SEED, help="Random seed.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation fraction taken from the train split.")
    parser.add_argument(
        "--pretrained",
        default="auto",
        choices=["auto", "default", "none"],
        help="Pretrained weight strategy for the transfer model.",
    )
    parser.add_argument(
        "--freeze_features",
        action="store_true",
        default=False,
        help="Freeze the convolutional feature extractor and train the classifier head only.",
    )
    parser.add_argument("--image_width", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Training image width.")
    parser.add_argument("--image_height", type=int, default=DEFAULT_IMAGE_SIZE[1], help="Training image height.")
    parser.add_argument(
        "--priority_loss_boost",
        type=float,
        default=2.5,
        help="Extra loss weight applied to the stain class during training.",
    )
    parser.add_argument(
        "--priority_sample_boost",
        type=float,
        default=2.0,
        help="Extra sampling weight applied to stain examples in the training loader.",
    )
    parser.add_argument(
        "--selection_macro_weight",
        type=float,
        default=1.0,
        help="Weight of macro F1 in the validation checkpoint-selection score.",
    )
    parser.add_argument(
        "--selection_priority_f1_weight",
        type=float,
        default=0.75,
        help="Weight of stain F1 in the validation checkpoint-selection score.",
    )
    parser.add_argument(
        "--selection_priority_recall_weight",
        type=float,
        default=0.5,
        help="Weight of stain recall in the validation checkpoint-selection score.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the transfer-learning model and save checkpoint artifacts."""
    args = parse_args()
    torch.manual_seed(args.seed)

    rows = load_metadata_rows_from_file(args.train_metadata)
    train_rows, val_rows = split_rows_by_label(rows, test_ratio=args.val_ratio, seed=args.seed)
    image_size = (args.image_width, args.image_height)
    priority_sampler = build_weighted_sampler(
        train_rows,
        priority_boost=args.priority_sample_boost,
    )

    train_loader = create_data_loader(
        rows=train_rows,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        train=True,
        num_workers=args.num_workers,
        sampler=priority_sampler,
    )

    device = torch.device("cpu")
    model, transfer_info = build_transfer_model(
        num_classes=len(CLASS_NAMES),
        pretrained_mode=args.pretrained,
        freeze_features=args.freeze_features,
    )
    model.to(device)

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=build_class_weight_tensor(
            train_rows,
            priority_boost=args.priority_loss_boost,
            device=device,
        )
    )

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_selection_score = -1.0
    best_macro_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion=criterion)
        val_metrics = evaluate_torch_model(
            model=model,
            rows=val_rows,
            data_dir=args.data_dir,
            device=device,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        selection_score = compute_priority_selection_score(
            val_metrics,
            macro_weight=args.selection_macro_weight,
            stain_f1_weight=args.selection_priority_f1_weight,
            stain_recall_weight=args.selection_priority_recall_weight,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_stain_recall": float(val_metrics["priority_class"]["recall"]),
            "val_stain_f1": float(val_metrics["priority_class"]["f1"]),
            "selection_score": float(selection_score),
        }
        history.append(epoch_record)

        if selection_score > best_selection_score:
            best_selection_score = float(selection_score)
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_state)

    model_dir = ensure_directory(args.model_dir)
    metrics_dir = ensure_directory(args.metrics_dir)
    checkpoint_path = model_dir / "main_model.pt"
    history_path = metrics_dir / "main_training_history.json"

    save_torch_checkpoint(
        checkpoint_path,
        {
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": list(image_size),
            "transfer_info": transfer_info,
            "tuning": {
                "priority_class": "stain",
                "priority_loss_boost": args.priority_loss_boost,
                "priority_sample_boost": args.priority_sample_boost,
                "selection_macro_weight": args.selection_macro_weight,
                "selection_priority_f1_weight": args.selection_priority_f1_weight,
                "selection_priority_recall_weight": args.selection_priority_recall_weight,
            },
            "data_marking": "SYNTHETIC",
        },
    )

    save_json(
        history_path,
        {
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "best_selection_score": best_selection_score,
            "history": history,
            "transfer_info": transfer_info,
            "tuning": {
                "priority_class": "stain",
                "priority_loss_boost": args.priority_loss_boost,
                "priority_sample_boost": args.priority_sample_boost,
                "selection_macro_weight": args.selection_macro_weight,
                "selection_priority_f1_weight": args.selection_priority_f1_weight,
                "selection_priority_recall_weight": args.selection_priority_recall_weight,
                "why": "Boost stain during training and checkpoint selection to reduce stain->acceptable confusion.",
            },
            "data_marking": "SYNTHETIC",
        },
    )

    print(f"Saved main model checkpoint to: {checkpoint_path}")
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()
