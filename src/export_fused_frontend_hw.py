"""
Export the current ECG frontend as fused Conv1d parameters plus classifier parameters.

This script fuses:
  Conv1d -> BatchNorm1d

into a single equivalent Conv1d, then exports:
  - fused conv weight and bias
  - classifier weight and bias
  - float text files
  - quantized .mem files
  - metadata JSON

Typical usage:
    python -m src.export_fused_frontend_hw \
      --checkpoint checkpoints/mitbih_lightweight.pt \
      --out-dir hardware_export/fused_frontend \
      --prefix ecg_fused_frontend
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.model import build_model_from_checkpoint, fuse_feature_extractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fused Conv+BN frontend parameters together with classifier parameters."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a full training checkpoint, for example checkpoints/mitbih_lightweight.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="hardware_export/fused_frontend",
        help="Directory for exported hardware files.",
    )
    parser.add_argument(
        "--prefix",
        default="ecg_fused_frontend",
        help="File prefix for generated outputs.",
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
        for row in matrix.reshape(matrix.shape[0], -1):
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

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint)
    model.eval()

    fused_conv_weight, fused_conv_bias = fuse_feature_extractor(model.feature_extractor)
    classifier = model.classifier

    conv_weight = fused_conv_weight.cpu().numpy().astype(np.float32)
    conv_bias = fused_conv_bias.cpu().numpy().astype(np.float32)
    classifier_weight = classifier.weight.detach().cpu().numpy().astype(np.float32)
    classifier_bias = classifier.bias.detach().cpu().numpy().astype(np.float32)

    conv_weight_flat = conv_weight.reshape(-1)
    classifier_weight_flat = classifier_weight.reshape(-1)

    q_conv_weight, conv_weight_clipped = quantize(conv_weight_flat, args.word_bits, args.frac_bits)
    q_conv_bias, conv_bias_clipped = quantize(conv_bias, args.word_bits, args.frac_bits)
    q_classifier_weight, classifier_weight_clipped = quantize(
        classifier_weight_flat,
        args.word_bits,
        args.frac_bits,
    )
    q_classifier_bias, classifier_bias_clipped = quantize(
        classifier_bias,
        args.word_bits,
        args.frac_bits,
    )

    stem = args.prefix
    paths = {
        "meta": out_dir / f"{stem}_meta.json",
        "conv_weight_float": out_dir / f"{stem}_conv_weight_float.txt",
        "conv_bias_float": out_dir / f"{stem}_conv_bias_float.txt",
        "conv_weight_mem": out_dir / f"{stem}_conv_weight_q{args.word_bits}_{args.frac_bits}.mem",
        "conv_bias_mem": out_dir / f"{stem}_conv_bias_q{args.word_bits}_{args.frac_bits}.mem",
        "classifier_weight_float": out_dir / f"{stem}_classifier_weight_float.txt",
        "classifier_bias_float": out_dir / f"{stem}_classifier_bias_float.txt",
        "classifier_weight_mem": out_dir
        / f"{stem}_classifier_weight_q{args.word_bits}_{args.frac_bits}.mem",
        "classifier_bias_mem": out_dir
        / f"{stem}_classifier_bias_q{args.word_bits}_{args.frac_bits}.mem",
    }

    conv = model.feature_extractor.conv
    meta = {
        "source_checkpoint": str(args.checkpoint),
        "frontend_type": "fused_conv1d_bn",
        "input_channels": int(conv.in_channels),
        "output_channels": int(conv.out_channels),
        "kernel_size": list(conv.kernel_size),
        "stride": list(conv.stride),
        "padding": list(conv.padding),
        "dilation": list(conv.dilation),
        "groups": int(conv.groups),
        "signal_length": int(checkpoint["signal_length"]),
        "feature_dim_after_pool_and_flatten": int(model.feature_dim),
        "classifier_weight_shape": list(classifier_weight.shape),
        "classifier_bias_shape": list(classifier_bias.shape),
        "word_bits": args.word_bits,
        "frac_bits": args.frac_bits,
        "scale": 1 << args.frac_bits,
        "conv_weight_clipped_values": conv_weight_clipped,
        "conv_bias_clipped_values": conv_bias_clipped,
        "classifier_weight_clipped_values": classifier_weight_clipped,
        "classifier_bias_clipped_values": classifier_bias_clipped,
        "hardware_flow": [
            "conv1d_with_fused_bn",
            "relu",
            "maxpool1d_kernel_2",
            "flatten",
            "linear_classifier",
        ],
    }

    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_text_matrix(paths["conv_weight_float"], conv_weight)
    save_text_vector(paths["conv_bias_float"], conv_bias)
    save_lines(paths["conv_weight_mem"], to_hex_lines(q_conv_weight, args.word_bits))
    save_lines(paths["conv_bias_mem"], to_hex_lines(q_conv_bias, args.word_bits))
    save_text_matrix(paths["classifier_weight_float"], classifier_weight)
    save_text_vector(paths["classifier_bias_float"], classifier_bias)
    save_lines(paths["classifier_weight_mem"], to_hex_lines(q_classifier_weight, args.word_bits))
    save_lines(paths["classifier_bias_mem"], to_hex_lines(q_classifier_bias, args.word_bits))

    print(f"Source checkpoint         : {args.checkpoint}")
    print(f"Fused conv weight shape   : {tuple(conv_weight.shape)}")
    print(f"Fused conv bias shape     : {tuple(conv_bias.shape)}")
    print(f"Classifier weight shape   : {tuple(classifier_weight.shape)}")
    print(f"Classifier bias shape     : {tuple(classifier_bias.shape)}")
    print(f"Fixed-point               : Q{args.word_bits - args.frac_bits - 1}.{args.frac_bits}")
    print("")
    print("Generated files:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
