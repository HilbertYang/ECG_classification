from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data_loader import create_dataloaders
from .model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample inference for ECG classification.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to the processed .npz dataset.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Which split to sample from.",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index inside the selected split.")
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

    bundle = create_dataloaders(
        dataset_path=dataset,
        use_demo=use_demo,
        demo_samples=config.get("demo_samples", 600),
        signal_length=checkpoint["signal_length"],
        batch_size=1,
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        normal_label=config.get("normal_label", 0),
        seed=config.get("seed", 42),
    )

    split_map = {
        "train": bundle.splits.train,
        "val": bundle.splits.val,
        "test": bundle.splits.test,
    }
    split = split_map[args.split]
    if len(split.labels) == 0:
        raise ValueError(f"The requested split '{args.split}' is empty.")
    if args.index < 0 or args.index >= len(split.labels):
        raise IndexError(
            f"Index {args.index} is out of range for split '{args.split}' "
            f"with {len(split.labels)} samples."
        )

    sample = torch.from_numpy(split.signals[args.index]).unsqueeze(0).to(device)
    target = int(split.labels[args.index])

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()

    with torch.no_grad():
        logits, features = model(sample, return_features=True)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        prediction = int(logits.argmax(dim=1).item())

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_source": bundle.source_name,
        "split": args.split,
        "index": args.index,
        "ground_truth": target,
        "prediction": prediction,
        "probabilities": probabilities,
        "feature_vector_shape": list(features.shape),
    }

    output_path = Path("results") / f"{args.checkpoint.stem}_{args.split}_sample_{args.index}.json"
    save_json(result, output_path)
    print(json.dumps(result, indent=2))
    print(f"Saved inference output to {output_path}")


if __name__ == "__main__":
    main()
