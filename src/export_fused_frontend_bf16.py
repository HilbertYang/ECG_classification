"""
Export the fused Conv1d+BatchNorm frontend and classifier parameters as BF16 files.

Typical usage:
    python -m src.export_fused_frontend_bf16 \
      --checkpoint checkpoints/mitbih_lightweight.pt \
      --out-dir hardware_export/fused_frontend_bf16 \
      --prefix ecg_fused_frontend
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.bf16_utils import bf16_hex_lines, round_to_bf16
from src.model import build_model_from_checkpoint, fuse_feature_extractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fused Conv+BN frontend parameters and classifier parameters as BF16."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a full training checkpoint, for example checkpoints/mitbih_lightweight.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="hardware_export/fused_frontend_bf16",
        help="Directory for exported files.",
    )
    parser.add_argument(
        "--prefix",
        default="ecg_fused_frontend",
        help="Prefix for generated files.",
    )
    return parser.parse_args()


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

    conv_weight = round_to_bf16(fused_conv_weight.cpu().numpy().astype(np.float32))
    conv_bias = round_to_bf16(fused_conv_bias.cpu().numpy().astype(np.float32))
    classifier_weight = round_to_bf16(
        classifier.weight.detach().cpu().numpy().astype(np.float32)
    )
    classifier_bias = round_to_bf16(
        classifier.bias.detach().cpu().numpy().astype(np.float32)
    )

    stem = args.prefix
    paths = {
        "meta": out_dir / f"{stem}_bf16_meta.json",
        "conv_weight_float": out_dir / f"{stem}_conv_weight_bf16_float.txt",
        "conv_bias_float": out_dir / f"{stem}_conv_bias_bf16_float.txt",
        "conv_weight_mem": out_dir / f"{stem}_conv_weight_bf16.mem",
        "conv_bias_mem": out_dir / f"{stem}_conv_bias_bf16.mem",
        "classifier_weight_float": out_dir / f"{stem}_classifier_weight_bf16_float.txt",
        "classifier_bias_float": out_dir / f"{stem}_classifier_bias_bf16_float.txt",
        "classifier_weight_mem": out_dir / f"{stem}_classifier_weight_bf16.mem",
        "classifier_bias_mem": out_dir / f"{stem}_classifier_bias_bf16.mem",
    }

    conv = model.feature_extractor.conv
    meta = {
        "source_checkpoint": str(args.checkpoint),
        "format": "bf16",
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
        "conv_weight_shape": list(conv_weight.shape),
        "conv_bias_shape": list(conv_bias.shape),
        "classifier_weight_shape": list(classifier_weight.shape),
        "classifier_bias_shape": list(classifier_bias.shape),
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
    save_lines(paths["conv_weight_mem"], bf16_hex_lines(conv_weight.reshape(-1)))
    save_lines(paths["conv_bias_mem"], bf16_hex_lines(conv_bias))
    save_text_matrix(paths["classifier_weight_float"], classifier_weight)
    save_text_vector(paths["classifier_bias_float"], classifier_bias)
    save_lines(paths["classifier_weight_mem"], bf16_hex_lines(classifier_weight.reshape(-1)))
    save_lines(paths["classifier_bias_mem"], bf16_hex_lines(classifier_bias))

    print(f"Source checkpoint       : {args.checkpoint}")
    print(f"Fused conv weight shape : {tuple(conv_weight.shape)}")
    print(f"Fused conv bias shape   : {tuple(conv_bias.shape)}")
    print(f"Classifier weight shape : {tuple(classifier_weight.shape)}")
    print(f"Classifier bias shape   : {tuple(classifier_bias.shape)}")
    print("Format                  : BF16")
    print("")
    print("Generated files:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
