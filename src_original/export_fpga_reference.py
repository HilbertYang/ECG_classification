"""
Export CNN feature vectors for FPGA input together with software reference outputs.

Typical usage:
    python -m src.export_fpga_reference \
      --checkpoint checkpoints/mitbih_baseline.pt \
      --dataset data/processed/mitbih_binary.npz \
      --split test \
      --max-samples 128 \
      --accumulation fp32 \
      --out-dir hardware_export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from src.bf16_utils import bf16_hex_lines, bf16_linear
from src.data_loader import create_dataloaders
from src.model import build_model_from_checkpoint
from src.preprocess import ensure_signal_shape, normalize_per_sample, resize_signals, to_binary_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CNN features for FPGA input and software reference outputs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the full training checkpoint.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to the processed .npz dataset.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split used for export.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Feature extraction batch size.")
    parser.add_argument(
        "--accumulation",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Accumulator behavior for the emulated classifier reference.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on exported samples.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("hardware_export"),
        help="Directory for exported FPGA input and reference files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix. Defaults to <checkpoint>_<split>.",
    )
    parser.add_argument(
        "--record-id",
        type=str,
        default=None,
        help="Optional MIT-BIH record id to export directly from dataset order, for example 100.",
    )
    parser.add_argument(
        "--first-n-beats",
        type=int,
        default=None,
        help="Optional number of beats to keep from the selected record in dataset order.",
    )
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_text_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in np.asarray(matrix, dtype=np.float32):
            handle.write(" ".join(f"{float(value):.10f}" for value in row))
            handle.write("\n")


def save_text_vector(path: Path, vector: np.ndarray) -> None:
    path.write_text(
        " ".join(str(int(value)) for value in np.asarray(vector).reshape(-1)) + "\n",
        encoding="utf-8",
    )


def save_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_record_subset(
    dataset_path: Path,
    record_id: str,
    first_n_beats: int | None,
    signal_length: int,
    normal_label: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(dataset_path, allow_pickle=False) as bundle:
        signals_key = "signals" if "signals" in bundle.files else "x"
        labels_key = "labels" if "labels" in bundle.files else "y"
        if signals_key not in bundle.files or labels_key not in bundle.files:
            raise KeyError("Dataset .npz file must contain either signals/labels or x/y arrays.")
        if "record_ids" not in bundle.files:
            raise KeyError("Dataset does not contain record_ids, so --record-id cannot be used.")

        signals = np.asarray(bundle[signals_key], dtype=np.float32)
        labels = np.asarray(bundle[labels_key]).reshape(-1)
        record_ids = np.asarray(bundle["record_ids"]).astype(str)
        beat_symbols = np.asarray(bundle["beat_symbols"]).astype(str) if "beat_symbols" in bundle.files else None
        beat_samples = np.asarray(bundle["beat_samples"], dtype=np.int64) if "beat_samples" in bundle.files else None

    mask = record_ids == str(record_id)
    if not np.any(mask):
        raise ValueError(f"Record id '{record_id}' was not found in dataset: {dataset_path}")

    signals = signals[mask]
    labels = labels[mask]
    record_ids = record_ids[mask]
    if beat_symbols is not None:
        beat_symbols = beat_symbols[mask]
    if beat_samples is not None:
        beat_samples = beat_samples[mask]

    if first_n_beats is not None:
        signals = signals[:first_n_beats]
        labels = labels[:first_n_beats]
        record_ids = record_ids[:first_n_beats]
        if beat_symbols is not None:
            beat_symbols = beat_symbols[:first_n_beats]
        if beat_samples is not None:
            beat_samples = beat_samples[:first_n_beats]

    signals = ensure_signal_shape(signals)
    signals = resize_signals(signals, signal_length)
    signals = normalize_per_sample(signals)
    labels = to_binary_labels(labels, normal_label=normal_label)

    metadata: dict[str, np.ndarray] = {
        "record_ids": record_ids,
    }
    if beat_symbols is not None:
        metadata["beat_symbols"] = beat_symbols
    if beat_samples is not None:
        metadata["beat_samples"] = beat_samples

    return signals.astype(np.float32), labels.astype(np.int64), metadata


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    dataset = args.dataset
    if dataset is None and config.get("dataset"):
        dataset = Path(config["dataset"])

    use_demo = args.demo or config.get("use_demo", False) or dataset is None
    export_metadata: dict[str, np.ndarray] = {}

    if args.record_id is not None:
        if dataset is None:
            raise ValueError("--record-id requires a real dataset path.")
        signals, labels, export_metadata = load_record_subset(
            dataset_path=dataset,
            record_id=args.record_id,
            first_n_beats=args.first_n_beats,
            signal_length=checkpoint["signal_length"],
            normal_label=config.get("normal_label", 0),
        )
        dataset_source = f"{dataset.stem}_record_{args.record_id}"
        selection_mode = "record_subset"
    else:
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
            "train": bundle.splits.train,
            "val": bundle.splits.val,
            "test": bundle.splits.test,
        }
        split = split_map[args.split]
        if len(split.labels) == 0:
            raise ValueError(f"The requested split '{args.split}' is empty.")

        signals = split.signals
        labels = split.labels.astype(np.int64)
        if args.max_samples is not None:
            signals = signals[: args.max_samples]
            labels = labels[: args.max_samples]
        dataset_source = bundle.source_name
        selection_mode = "split"

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    classifier = model.classifier
    weight = classifier.weight.detach().cpu().numpy().astype(np.float32)
    bias = classifier.bias.detach().cpu().numpy().astype(np.float32)

    feature_batches: list[np.ndarray] = []
    fp32_logits_batches: list[np.ndarray] = []
    bf16_logits_batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(labels), args.batch_size):
            stop = min(start + args.batch_size, len(labels))
            batch_signals = torch.from_numpy(signals[start:stop]).to(device=device, dtype=torch.float32)
            features = model.extract_features(batch_signals).cpu().numpy().astype(np.float32)
            fp32_logits = classifier(torch.from_numpy(features).to(device)).cpu().numpy().astype(np.float32)
            bf16_logits, _, _, _, _ = bf16_linear(
                features=features,
                weight=weight,
                bias=bias,
                accumulation=args.accumulation,
            )

            feature_batches.append(features)
            fp32_logits_batches.append(fp32_logits)
            bf16_logits_batches.append(bf16_logits.astype(np.float32))

    features = np.concatenate(feature_batches, axis=0)
    fp32_logits = np.concatenate(fp32_logits_batches, axis=0)
    bf16_logits = np.concatenate(bf16_logits_batches, axis=0)
    fp32_pred = fp32_logits.argmax(axis=1).astype(np.int64)
    bf16_pred = bf16_logits.argmax(axis=1).astype(np.int64)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.record_id is not None:
        beat_count_suffix = f"_{labels.shape[0]}beats"
        default_prefix = f"{args.checkpoint.stem}_record_{args.record_id}{beat_count_suffix}"
    else:
        default_prefix = f"{args.checkpoint.stem}_{args.split}"
    stem = args.prefix or default_prefix

    meta_path = out_dir / f"{stem}_meta.json"
    features_txt_path = out_dir / f"{stem}_features_fp32.txt"
    features_bf16_mem_path = out_dir / f"{stem}_features_bf16.mem"
    labels_path = out_dir / f"{stem}_labels.txt"
    fp32_logits_path = out_dir / f"{stem}_fp32_logits.txt"
    bf16_logits_path = out_dir / f"{stem}_bf16_logits.txt"
    fp32_pred_path = out_dir / f"{stem}_fp32_pred.txt"
    bf16_pred_path = out_dir / f"{stem}_bf16_pred.txt"
    bundle_path = out_dir / f"{stem}_reference_bundle.npz"
    record_ids_path = out_dir / f"{stem}_record_ids.txt"
    beat_symbols_path = out_dir / f"{stem}_beat_symbols.txt"
    beat_samples_path = out_dir / f"{stem}_beat_samples.txt"

    save_text_matrix(features_txt_path, features)
    save_lines(features_bf16_mem_path, bf16_hex_lines(features.reshape(-1)))
    save_text_vector(labels_path, labels)
    save_text_matrix(fp32_logits_path, fp32_logits)
    save_text_matrix(bf16_logits_path, bf16_logits)
    save_text_vector(fp32_pred_path, fp32_pred)
    save_text_vector(bf16_pred_path, bf16_pred)

    bundle_payload: dict[str, np.ndarray] = {
        "features": features.astype(np.float32),
        "labels": labels.astype(np.int64),
        "fp32_logits": fp32_logits.astype(np.float32),
        "bf16_logits": bf16_logits.astype(np.float32),
        "fp32_pred": fp32_pred.astype(np.int64),
        "bf16_pred": bf16_pred.astype(np.int64),
    }

    if "record_ids" in export_metadata:
        record_ids = export_metadata["record_ids"].astype(str)
        record_ids_path.write_text("\n".join(record_ids.tolist()) + "\n", encoding="utf-8")
        bundle_payload["record_ids"] = record_ids
    if "beat_symbols" in export_metadata:
        beat_symbols = export_metadata["beat_symbols"].astype(str)
        beat_symbols_path.write_text("\n".join(beat_symbols.tolist()) + "\n", encoding="utf-8")
        bundle_payload["beat_symbols"] = beat_symbols
    if "beat_samples" in export_metadata:
        beat_samples = export_metadata["beat_samples"].astype(np.int64)
        save_text_vector(beat_samples_path, beat_samples)
        bundle_payload["beat_samples"] = beat_samples

    np.savez(bundle_path, **bundle_payload)

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_source": dataset_source,
        "split": args.split,
        "selection_mode": selection_mode,
        "record_id": args.record_id,
        "first_n_beats": args.first_n_beats,
        "samples_exported": int(labels.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_classes": int(fp32_logits.shape[1]),
        "accumulation": args.accumulation,
        "fp32_accuracy": float(accuracy_score(labels, fp32_pred)),
        "bf16_accuracy": float(accuracy_score(labels, bf16_pred)),
        "prediction_agreement": float(np.mean(fp32_pred == bf16_pred)),
        "mean_abs_logit_diff": float(np.abs(fp32_logits - bf16_logits).mean()),
        "max_abs_logit_diff": float(np.abs(fp32_logits - bf16_logits).max()),
        "files": {
            "features_fp32_txt": str(features_txt_path),
            "features_bf16_mem": str(features_bf16_mem_path),
            "labels_txt": str(labels_path),
            "fp32_logits_txt": str(fp32_logits_path),
            "bf16_logits_txt": str(bf16_logits_path),
            "fp32_pred_txt": str(fp32_pred_path),
            "bf16_pred_txt": str(bf16_pred_path),
            "reference_bundle_npz": str(bundle_path),
        },
    }
    if "record_ids" in export_metadata:
        result["files"]["record_ids_txt"] = str(record_ids_path)
    if "beat_symbols" in export_metadata:
        result["files"]["beat_symbols_txt"] = str(beat_symbols_path)
    if "beat_samples" in export_metadata:
        result["files"]["beat_samples_txt"] = str(beat_samples_path)
    save_json(result, meta_path)

    print(json.dumps(result, indent=2))
    print(f"Saved FPGA reference export to {out_dir}")


if __name__ == "__main__":
    main()
