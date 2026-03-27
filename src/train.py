from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .data_loader import create_dataloaders
from .model import ECGNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline ECG classifier.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to a processed .npz dataset.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    parser.add_argument("--demo-samples", type=int, default=600, help="Number of demo samples to generate.")
    parser.add_argument("--run-name", type=str, default="baseline", help="Checkpoint and result file prefix.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--signal-length", type=int, default=256, help="Target ECG segment length.")
    parser.add_argument(
        "--hidden-channels",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Conv channel sizes for the feature extractor.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--normal-label", type=int, default=0, help="Label value mapped to class 0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(signals)
            loss = criterion(logits, labels)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_demo = args.demo or args.dataset is None

    bundle = create_dataloaders(
        dataset_path=args.dataset,
        use_demo=use_demo,
        demo_samples=args.demo_samples,
        signal_length=args.signal_length,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        normal_label=args.normal_label,
        seed=args.seed,
    )

    model = ECGNet(
        input_channels=bundle.input_channels,
        signal_length=bundle.signal_length,
        num_classes=bundle.num_classes,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float | int]] = []
    best_val_accuracy = float("-inf")
    best_epoch = -1

    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.run_name}.pt"

    config = {
        "dataset": str(args.dataset) if args.dataset else None,
        "use_demo": use_demo,
        "demo_samples": args.demo_samples,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "signal_length": args.signal_length,
        "hidden_channels": list(args.hidden_channels),
        "dropout": args.dropout,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "normal_label": args.normal_label,
        "seed": args.seed,
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, bundle.train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, bundle.val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "input_channels": bundle.input_channels,
                    "signal_length": bundle.signal_length,
                    "num_classes": bundle.num_classes,
                    "source_name": bundle.source_name,
                },
                checkpoint_path,
            )

    summary = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "source_name": bundle.source_name,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
        "feature_vector_length": model.feature_dim,
    }
    save_json(summary, results_dir / f"{args.run_name}_train.json")

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
    print(f"Classifier input feature length: {model.feature_dim}")


if __name__ == "__main__":
    main()
