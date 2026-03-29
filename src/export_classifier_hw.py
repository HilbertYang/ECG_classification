"""
Export the trained classifier into files that are easier to load into FPGA logic.

Typical usage:
    python -m src.export_classifier_hw \
      --classifier-checkpoint checkpoints/mitbih_baseline_classifier.pt \
      --out-dir hardware_export \
      --prefix netfpga_classifier

The exporter writes:
  - a metadata JSON file
  - a human-readable float weight matrix
  - a human-readable float bias vector
  - a quantized .mem file for weights
  - a quantized .mem file for bias

The .mem files use signed two's-complement hex words and can be fed into
custom memory loaders or FPGA memory-init flows after you match the exact
layout expected by your NetFPGA design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export classifier parameters for FPGA use.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint",
        help="Path to the full training checkpoint, for example checkpoints/mitbih_baseline.pt",
    )
    group.add_argument(
        "--classifier-checkpoint",
        help="Path to the split classifier checkpoint, for example checkpoints/mitbih_baseline_classifier.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="hardware_export",
        help="Directory for exported hardware files.",
    )
    parser.add_argument(
        "--prefix",
        default="classifier_hw",
        help="File prefix for generated outputs.",
    )
    parser.add_argument(
        "--layout",
        choices=["row-major", "col-major"],
        default="row-major",
        help=(
            "Memory layout for the flattened weight stream. "
            "row-major stores all inputs for output 0 first; "
            "col-major stores all outputs for input 0 first."
        ),
    )
    parser.add_argument(
        "--word-bits",
        type=int,
        default=16,
        help="Signed fixed-point word width.",
    )
    parser.add_argument(
        "--frac-bits",
        type=int,
        default=8,
        help="Number of fractional bits for fixed-point export.",
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
        weight = classifier.weight.detach().cpu().numpy().astype(np.float32)
        bias = classifier.bias.detach().cpu().numpy().astype(np.float32)
        source = str(checkpoint_path)
        return weight, bias, source

    state_dict = torch.load(classifier_checkpoint_path, map_location="cpu")
    if "weight" not in state_dict or "bias" not in state_dict:
        raise KeyError("Classifier checkpoint must contain 'weight' and 'bias'.")

    weight = state_dict["weight"].detach().cpu().numpy().astype(np.float32)
    bias = state_dict["bias"].detach().cpu().numpy().astype(np.float32)
    source = str(classifier_checkpoint_path)
    return weight, bias, source


def flatten_weight(weight: np.ndarray, layout: str) -> np.ndarray:
    if layout == "row-major":
        return weight.reshape(-1)
    if layout == "col-major":
        return weight.transpose(1, 0).reshape(-1)
    raise ValueError(f"Unsupported layout: {layout}")


def quantize(values: np.ndarray, word_bits: int, frac_bits: int) -> tuple[np.ndarray, int]:
    scale = 1 << frac_bits
    min_int = -(1 << (word_bits - 1))
    max_int = (1 << (word_bits - 1)) - 1
    quantized = np.rint(values * scale).astype(np.int64)
    clipped = np.clip(quantized, min_int, max_int)
    clipped_count = int(np.count_nonzero(quantized != clipped))
    return clipped, clipped_count


def to_hex_lines(values: np.ndarray, word_bits: int) -> list[str]:
    width = (word_bits + 3) // 4
    mask = (1 << word_bits) - 1
    return [format(int(value) & mask, f"0{width}X") for value in values.reshape(-1)]


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

    flat_weight = flatten_weight(weight, args.layout)
    q_weight, weight_clipped = quantize(flat_weight, args.word_bits, args.frac_bits)
    q_bias, bias_clipped = quantize(bias, args.word_bits, args.frac_bits)

    meta = {
        "source_checkpoint": source,
        "input_dim": int(weight.shape[1]),
        "output_dim": int(weight.shape[0]),
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "formula": "logits[o] = sum_i weight[o][i] * features[i] + bias[o]",
        "weight_layout": args.layout,
        "word_bits": args.word_bits,
        "frac_bits": args.frac_bits,
        "scale": 1 << args.frac_bits,
        "weight_clipped_values": weight_clipped,
        "bias_clipped_values": bias_clipped,
    }

    stem = args.prefix
    meta_path = out_dir / f"{stem}_meta.json"
    weight_matrix_path = out_dir / f"{stem}_weight_matrix.txt"
    bias_float_path = out_dir / f"{stem}_bias_float.txt"
    weight_mem_path = out_dir / f"{stem}_weight_q{args.word_bits}_{args.frac_bits}.mem"
    bias_mem_path = out_dir / f"{stem}_bias_q{args.word_bits}_{args.frac_bits}.mem"

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_text_matrix(weight_matrix_path, weight)
    save_text_vector(bias_float_path, bias)
    save_lines(weight_mem_path, to_hex_lines(q_weight, args.word_bits))
    save_lines(bias_mem_path, to_hex_lines(q_bias, args.word_bits))

    print(f"Source checkpoint : {source}")
    print(f"Weight shape      : {tuple(weight.shape)}")
    print(f"Bias shape        : {tuple(bias.shape)}")
    print(f"Layout            : {args.layout}")
    print(f"Fixed-point       : Q{args.word_bits - args.frac_bits - 1}.{args.frac_bits}")
    print(f"Weight clips      : {weight_clipped}")
    print(f"Bias clips        : {bias_clipped}")
    print("")
    print("Generated files:")
    print(f"  {meta_path}")
    print(f"  {weight_matrix_path}")
    print(f"  {bias_float_path}")
    print(f"  {weight_mem_path}")
    print(f"  {bias_mem_path}")


if __name__ == "__main__":
    main()
