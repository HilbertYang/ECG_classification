"""
Emulate the classifier head in BF16 so deployment can be checked before hardware mapping.

Example:
    python -m src.emulate_classifier_bf16 \
      --checkpoint checkpoints/mitbih_baseline.pt \
      --dataset data/processed/mitbih_binary.npz \
      --split test \
      --accumulation fp32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from src.bf16_utils import bf16_linear
from src.data_loader import create_dataloaders
from src.model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FP32 classifier outputs against BF16 emulation.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the full training checkpoint.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to the processed .npz dataset.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split used for comparison.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument(
        "--accumulation",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Accumulator behavior for the emulated classifier.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to compare.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path. Defaults to results/<checkpoint>_<split>_bf16_<accumulation>.json",
    )
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
        batch_size=args.batch_size,
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
    classifier = model.classifier
    weight = classifier.weight.detach().cpu().numpy().astype(np.float32)
    bias = classifier.bias.detach().cpu().numpy().astype(np.float32)

    all_labels: list[np.ndarray] = []
    all_fp32_logits: list[np.ndarray] = []
    all_bf16_logits: list[np.ndarray] = []
    processed_samples = 0

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            features = model.extract_features(signals).cpu().numpy().astype(np.float32)
            fp32_logits = classifier(torch.from_numpy(features).to(device)).cpu().numpy().astype(np.float32)
            bf16_logits, _, _, _, _ = bf16_linear(
                features=features,
                weight=weight,
                bias=bias,
                accumulation=args.accumulation,
            )

            if args.max_samples is not None:
                remaining = args.max_samples - processed_samples
                if remaining <= 0:
                    break
                fp32_logits = fp32_logits[:remaining]
                bf16_logits = bf16_logits[:remaining]
                labels = labels[:remaining]

            all_fp32_logits.append(fp32_logits)
            all_bf16_logits.append(bf16_logits)
            all_labels.append(labels.numpy())
            processed_samples += labels.shape[0]

            if args.max_samples is not None and processed_samples >= args.max_samples:
                break

    y_true = np.concatenate(all_labels)
    fp32_logits = np.concatenate(all_fp32_logits)
    bf16_logits = np.concatenate(all_bf16_logits)
    fp32_pred = fp32_logits.argmax(axis=1)
    bf16_pred = bf16_logits.argmax(axis=1)
    logit_diff = np.abs(fp32_logits - bf16_logits)

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_source": bundle.source_name,
        "split": args.split,
        "samples_compared": int(y_true.shape[0]),
        "accumulation": args.accumulation,
        "fp32_accuracy": float(accuracy_score(y_true, fp32_pred)),
        "bf16_accuracy": float(accuracy_score(y_true, bf16_pred)),
        "prediction_agreement": float(np.mean(fp32_pred == bf16_pred)),
        "mean_abs_logit_diff": float(logit_diff.mean()),
        "max_abs_logit_diff": float(logit_diff.max()),
    }

    if args.output is None:
        output_path = Path("results") / (
            f"{args.checkpoint.stem}_{args.split}_bf16_{args.accumulation}.json"
        )
    else:
        output_path = args.output

    save_json(result, output_path)
    print(json.dumps(result, indent=2))
    print(f"Saved BF16 emulation report to {output_path}")


if __name__ == "__main__":
    main()
