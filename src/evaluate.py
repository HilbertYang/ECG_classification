from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data_loader import create_dataloaders
from .model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ECG classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to the processed .npz dataset.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    dataset = args.dataset
    if dataset is None and config.get("dataset"):
        dataset = Path(config["dataset"])

    use_demo = args.demo or config.get("use_demo", False) or dataset is None
    batch_size = args.batch_size or config.get("batch_size", 64)

    bundle = create_dataloaders(
        dataset_path=dataset,
        use_demo=use_demo,
        demo_samples=config.get("demo_samples", 600),
        signal_length=checkpoint["signal_length"],
        batch_size=batch_size,
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        normal_label=config.get("normal_label", 0),
        seed=config.get("seed", 42),
    )

    split_map = {
        "train": bundle.train_loader,
        "val": bundle.val_loader,
        "test": bundle.test_loader,
    }
    loader = split_map[args.split]
    if len(loader.dataset) == 0:
        raise ValueError(f"The requested split '{args.split}' is empty.")

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()

    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for signals, labels in loader:
            logits = model(signals.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "dataset_source": bundle.source_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    output_path = Path("results") / f"{args.checkpoint.stem}_{args.split}_metrics.json"
    save_json(metrics, output_path)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
