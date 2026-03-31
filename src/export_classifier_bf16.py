"""
Export the classifier head as BF16-encoded memory files for hardware loading.

Example:
    python -m src.export_classifier_bf16 \
      --classifier-checkpoint checkpoints/mitbih_baseline_classifier.pt \
      --out-dir hardware_export \
      --prefix netfpga_classifier
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.bf16_utils import bf16_hex_lines, flatten_weight, round_to_bf16
from src.model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export classifier parameters as BF16 memory files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Path to a full training checkpoint.")
    group.add_argument("--classifier-checkpoint", help="Path to a split classifier checkpoint.")
    parser.add_argument("--out-dir", default="hardware_export", help="Directory for exported files.")
    parser.add_argument("--prefix", default="classifier_bf16", help="Prefix for generated files.")
    parser.add_argument(
        "--layout",
        choices=["row-major", "col-major"],
        default="row-major",
        help="Flattened weight layout expected by the hardware memory interface.",
    )
    return parser.parse_args()


def load_classifier_arrays(
    checkpoint_path: str | None,
    classifier_checkpoint_path: str | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model = build_model_from_checkpoint(ckpt)
        classifier = model.classifier
        return (
            classifier.weight.detach().cpu().numpy().astype(np.float32),
            classifier.bias.detach().cpu().numpy().astype(np.float32),
            str(checkpoint_path),
        )

    state_dict = torch.load(classifier_checkpoint_path, map_location="cpu")
    if "weight" not in state_dict or "bias" not in state_dict:
        raise KeyError("Classifier checkpoint must contain 'weight' and 'bias'.")
    return (
        state_dict["weight"].detach().cpu().numpy().astype(np.float32),
        state_dict["bias"].detach().cpu().numpy().astype(np.float32),
        str(classifier_checkpoint_path),
    )


def save_text_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(" ".join(f"{float(value):.10f}" for value in row))
            handle.write("\n")


def save_text_vector(path: Path, vector: np.ndarray) -> None:
    path.write_text(
        " ".join(f"{float(value):.10f}" for value in vector.reshape(-1)) + "\n",
        encoding="utf-8",
    )


def save_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight, bias, source = load_classifier_arrays(
        checkpoint_path=args.checkpoint,
        classifier_checkpoint_path=args.classifier_checkpoint,
    )

    if weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("Expected classifier weight to be 2D and bias to be 1D.")
    if weight.shape[0] != bias.shape[0]:
        raise ValueError("Bias length must match the classifier output dimension.")

    weight_bf16 = round_to_bf16(weight)
    bias_bf16 = round_to_bf16(bias)
    flat_weight = flatten_weight(weight_bf16, args.layout)

    meta = {
        "source_checkpoint": source,
        "format": "bf16",
        "input_dim": int(weight.shape[1]),
        "output_dim": int(weight.shape[0]),
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "weight_layout": args.layout,
        "formula": "logits[o] = sum_i weight[o][i] * features[i] + bias[o]",
    }

    stem = args.prefix
    meta_path = out_dir / f"{stem}_bf16_meta.json"
    weight_float_path = out_dir / f"{stem}_weight_bf16_float.txt"
    bias_float_path = out_dir / f"{stem}_bias_bf16_float.txt"
    weight_mem_path = out_dir / f"{stem}_weight_bf16.mem"
    bias_mem_path = out_dir / f"{stem}_bias_bf16.mem"

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_text_matrix(weight_float_path, weight_bf16)
    save_text_vector(bias_float_path, bias_bf16)
    save_lines(weight_mem_path, bf16_hex_lines(flat_weight))
    save_lines(bias_mem_path, bf16_hex_lines(bias_bf16))

    print(f"Source checkpoint : {source}")
    print(f"Weight shape      : {tuple(weight.shape)}")
    print(f"Bias shape        : {tuple(bias.shape)}")
    print(f"Layout            : {args.layout}")
    print("Format            : BF16")
    print("")
    print("Generated files:")
    print(f"  {meta_path}")
    print(f"  {weight_float_path}")
    print(f"  {bias_float_path}")
    print(f"  {weight_mem_path}")
    print(f"  {bias_mem_path}")


if __name__ == "__main__":
    main()
