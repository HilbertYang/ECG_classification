"""
Export model input windows as BF16 files for hardware input testing.

Typical usage:
    python -m src.export_input_bf16 \
      --checkpoint checkpoints/mitbih_lightweight.pt \
      --dataset data/processed/mitbih_binary.npz \
      --record-id 200 \
      --first-n-beats 4 \
      --out-dir hardware_export/input_bf16 \
      --prefix record200_4beats_input
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.bf16_utils import bf16_hex_lines
from src.preprocess import ensure_signal_shape, normalize_per_sample, resize_signals, to_binary_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw model input windows as BF16.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the full training checkpoint.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the processed .npz dataset.")
    parser.add_argument("--record-id", type=str, required=True, help="MIT-BIH record id, for example 200.")
    parser.add_argument("--first-n-beats", type=int, default=None, help="Optional number of beats to export.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("hardware_export/input_bf16"),
        help="Directory for exported files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix. Defaults to <checkpoint>_record_<id>_<n>beats_input.",
    )
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_text_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in np.asarray(matrix, dtype=np.float32).reshape(matrix.shape[0], -1):
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

    metadata: dict[str, np.ndarray] = {"record_ids": record_ids}
    if beat_symbols is not None:
        metadata["beat_symbols"] = beat_symbols
    if beat_samples is not None:
        metadata["beat_samples"] = beat_samples

    return signals.astype(np.float32), labels.astype(np.int64), metadata


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})

    signals, labels, metadata = load_record_subset(
        dataset_path=args.dataset,
        record_id=args.record_id,
        first_n_beats=args.first_n_beats,
        signal_length=checkpoint["signal_length"],
        normal_label=config.get("normal_label", 0),
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    beat_count_suffix = f"_{signals.shape[0]}beats"
    default_prefix = f"{args.checkpoint.stem}_record_{args.record_id}{beat_count_suffix}_input"
    stem = args.prefix or default_prefix

    flattened_signals = signals.reshape(signals.shape[0], -1)

    files = {
        "inputs_fp32_txt": out_dir / f"{stem}_fp32.txt",
        "inputs_bf16_mem": out_dir / f"{stem}_bf16.mem",
        "labels_txt": out_dir / f"{stem}_labels.txt",
        "record_ids_txt": out_dir / f"{stem}_record_ids.txt",
        "beat_symbols_txt": out_dir / f"{stem}_beat_symbols.txt",
        "beat_samples_txt": out_dir / f"{stem}_beat_samples.txt",
        "meta_json": out_dir / f"{stem}_meta.json",
    }

    save_text_matrix(files["inputs_fp32_txt"], flattened_signals)
    save_lines(files["inputs_bf16_mem"], bf16_hex_lines(flattened_signals.reshape(-1)))
    save_text_vector(files["labels_txt"], labels)
    save_lines(files["record_ids_txt"], metadata["record_ids"].astype(str).tolist())
    if "beat_symbols" in metadata:
        save_lines(files["beat_symbols_txt"], metadata["beat_symbols"].astype(str).tolist())
    if "beat_samples" in metadata:
        save_lines(files["beat_samples_txt"], [str(int(v)) for v in metadata["beat_samples"]])

    meta = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "record_id": str(args.record_id),
        "first_n_beats": args.first_n_beats,
        "samples_exported": int(signals.shape[0]),
        "input_shape": list(signals.shape),
        "flattened_input_shape": list(flattened_signals.shape),
        "signal_length": int(checkpoint["signal_length"]),
        "format": "bf16",
        "files": {key: str(value) for key, value in files.items()},
    }
    save_json(meta, files["meta_json"])

    print(f"Source checkpoint : {args.checkpoint}")
    print(f"Dataset           : {args.dataset}")
    print(f"Input shape       : {tuple(signals.shape)}")
    print(f"Flattened shape   : {tuple(flattened_signals.shape)}")
    print("Format            : BF16")
    print("")
    print("Generated files:")
    for path in files.values():
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()
