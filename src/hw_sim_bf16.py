#!/usr/bin/env python3
"""
Pure Python BF16 hardware simulation of the ECG classifier.
No PyTorch / NumPy used.

Pipeline (mirrors CLAUDE.md hardware_flow):
    input (4, 1, 256)
    -> Conv1d (kernel=7, padding=3, fused BN bias)
    -> ReLU
    -> MaxPool1d (kernel=2, stride=2)
    -> Flatten  -> (128,)
    -> Linear   (128 -> 2)

BF16 arithmetic rules applied throughout:
  - bf16_mul  : truncate product  to BF16
  - bf16_add  : truncate sum      to BF16
  - bf16_mac  : acc = bf16_add(acc, bf16_mul(a, b))
  - All intermediate values are BF16 floats (Python float truncated
    to 16-bit BF16 precision, stored as 32-bit Python float).

Usage:
    python3 -m src.hw_sim_bf16

or with custom paths:
    python3 -m src.hw_sim_bf16 \
        --fp-dir hardware_export/fused_frontend_bf16 \
        --in-dir hardware_export/input_bf16 \
        --ref-dir hardware_export/mini_feature \
        --prefix mini_mitbih_record_200_4beats
"""

import struct
import argparse
import os

# ============================================================
# BF16 primitives
# ============================================================

def fp32_to_bf16(f: float) -> float:
    """
    Truncate an IEEE 754 single-precision float to BF16 precision.
    BF16 keeps the top 16 bits of FP32 (1 sign + 8 exponent + 7 mantissa).
    The lower 16 mantissa bits are zeroed (truncation, not round-to-nearest).
    NaN and Inf pass through unchanged.
    """
    try:
        raw = struct.pack('>f', f)
    except (struct.error, OverflowError):
        return f
    # Zero out the lower 2 bytes → BF16 precision
    return struct.unpack('>f', raw[:2] + b'\x00\x00')[0]


def bf16_hex_to_float(h: str) -> float:
    """
    Convert a 4-character big-endian hex BF16 literal (e.g. '3F04') to float.
    Pads two zero bytes on the right to form a valid FP32 word.
    """
    return struct.unpack('>f', bytes.fromhex(h + '0000'))[0]


def float_to_bf16_hex(f: float) -> str:
    """Convert a Python float to its 4-char BF16 hex representation."""
    raw = struct.pack('>f', fp32_to_bf16(f))
    return raw[:2].hex().upper()


def bf16_mul(a: float, b: float) -> float:
    """
    BF16 multiply.
    Inputs are assumed to already be BF16-precision floats.
    Result is truncated to BF16.
    """
    return fp32_to_bf16(a * b)


def bf16_add(a: float, b: float) -> float:
    """
    BF16 add.
    Result is truncated to BF16.
    """
    return fp32_to_bf16(a + b)


def bf16_mac(acc: float, a: float, b: float) -> float:
    """
    BF16 multiply-accumulate: acc = bf16_add(acc, bf16_mul(a, b)).
    Models a single-cycle MAC unit that truncates both the product
    and the accumulated sum to BF16 before writing back.
    """
    return bf16_add(acc, bf16_mul(a, b))


# ============================================================
# File I/O helpers
# ============================================================

def load_bf16_mem(filepath: str) -> list:
    """
    Load a .mem file where each non-empty line is a 4-char BF16 hex word.
    Returns a list of Python floats at BF16 precision.
    """
    values = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if line:
                values.append(bf16_hex_to_float(line))
    return values


def load_fp32_logits_txt(filepath: str) -> list:
    """Load a text file where each line is a space-separated pair of logits."""
    rows = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split()])
    return rows


def load_fp32_features_txt(filepath: str) -> list:
    """Load a text file where each line is space-separated feature values."""
    rows = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split()])
    return rows


# ============================================================
# Hardware pipeline stages
# ============================================================

def conv1d_bf16(signal: list, weights: list, bias: float,
                kernel_size: int = 7, padding: int = 3) -> list:
    """
    1-D convolution with BF16 MAC accumulation.

    Parameters
    ----------
    signal      : input samples (length L), BF16 floats
    weights     : kernel coefficients (length kernel_size), BF16 floats
    bias        : scalar bias (BF16), added after the dot-product
    kernel_size : number of taps (default 7)
    padding     : zero-padding on each side (default 3 → same-size output)

    Output length = L  (stride=1, dilation=1)

    BF16 flow per output sample i:
        acc = 0.0  (BF16 zero)
        for k in 0..kernel_size-1:
            pos = i + k - padding
            x   = signal[pos] if 0 <= pos < L else 0.0
            acc = bf16_mac(acc, x, weights[k])
        out[i] = bf16_add(acc, bias)
    """
    L = len(signal)
    out = []
    for i in range(L):
        acc = fp32_to_bf16(0.0)          # BF16 accumulator starts at 0
        for k in range(kernel_size):
            pos = i + k - padding
            x = signal[pos] if 0 <= pos < L else 0.0
            acc = bf16_mac(acc, x, weights[k])
        out.append(bf16_add(acc, bias))
    return out


def relu_bf16(signal: list) -> list:
    """
    ReLU activation: max(0, x).
    Pure comparison — no arithmetic rounding involved.
    """
    return [x if x > 0.0 else 0.0 for x in signal]


def maxpool1d_bf16(signal: list, kernel_size: int = 2) -> list:
    """
    1-D max-pooling with stride = kernel_size (non-overlapping windows).
    Each window of kernel_size elements emits the maximum.

    With kernel_size=2 and input length 256 → output length 128.
    Selection is exact (comparison only, no arithmetic).
    """
    out = []
    for i in range(0, len(signal) - kernel_size + 1, kernel_size):
        window = signal[i : i + kernel_size]
        out.append(max(window))
    return out


def linear_bf16(features: list, weights_2d: list, bias: list) -> list:
    """
    Fully-connected (linear) layer with BF16 MAC accumulation.

    Parameters
    ----------
    features    : input vector (length 128), BF16 floats
    weights_2d  : list of num_classes rows, each of length 128, BF16 floats
    bias        : list of num_classes bias scalars, BF16 floats

    BF16 flow per output neuron j:
        acc = 0.0
        for i in 0..127:
            acc = bf16_mac(acc, features[i], weights_2d[j][i])
        logit[j] = bf16_add(acc, bias[j])
    """
    logits = []
    for j, row in enumerate(weights_2d):
        acc = fp32_to_bf16(0.0)
        for w, x in zip(row, features):
            acc = bf16_mac(acc, w, x)
        logits.append(bf16_add(acc, bias[j]))
    return logits


# ============================================================
# Full pipeline
# ============================================================

def run_pipeline(signal: list, conv_w: list, conv_b: float,
                 cls_w: list, cls_b: list, verbose: bool = False) -> dict:
    """
    Run one beat through the full hardware pipeline.

    Returns a dict with the output of every stage.
    """
    # Stage 1 — Fused Conv1d + BN bias
    conv_out = conv1d_bf16(signal, conv_w, conv_b)

    # Stage 2 — ReLU
    relu_out = relu_bf16(conv_out)

    # Stage 3 — MaxPool1d(2)
    pool_out = maxpool1d_bf16(relu_out, kernel_size=2)

    # Stage 4 — Flatten  (already a flat list of 128 values)
    features = pool_out

    # Stage 5 — Linear classifier
    logits = linear_bf16(features, cls_w, cls_b)

    return {
        "conv":     conv_out,
        "relu":     relu_out,
        "pool":     pool_out,
        "features": features,   # alias for pool output
        "logits":   logits,
        "pred":     0 if logits[0] > logits[1] else 1,
    }


# ============================================================
# Comparison helpers
# ============================================================

def max_abs_diff(a: list, b: list) -> float:
    return max(abs(x - y) for x, y in zip(a, b))

def mean_abs_diff(a: list, b: list) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pure Python BF16 hardware simulation of ECG classifier"
    )
    parser.add_argument(
        "--fp-dir",
        default="hardware_export/fused_frontend_bf16",
        help="Directory containing fused BF16 parameter .mem files",
    )
    parser.add_argument(
        "--in-dir",
        default="hardware_export/input_bf16",
        help="Directory containing input BF16 .mem file",
    )
    parser.add_argument(
        "--ref-dir",
        default="hardware_export/mini_feature",
        help="Directory containing golden-reference output files",
    )
    parser.add_argument(
        "--in-prefix",
        default="mini_mitbih_record_200_4beats_input",
        help="Prefix for input files",
    )
    parser.add_argument(
        "--ref-prefix",
        default="mini_mitbih_record_200_4beats",
        help="Prefix for reference output files",
    )
    parser.add_argument(
        "--num-beats", type=int, default=4,
        help="Number of beats in the input file",
    )
    parser.add_argument(
        "--signal-len", type=int, default=256,
        help="Samples per beat",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print first 16 values of each intermediate stage",
    )
    args = parser.parse_args()

    FP = args.fp_dir
    ID = args.in_dir
    RD = args.ref_dir
    IP = args.in_prefix
    RP = args.ref_prefix

    # ----------------------------------------------------------
    # Load parameters
    # ----------------------------------------------------------
    conv_weights = load_bf16_mem(os.path.join(FP, "ecg_fused_frontend_conv_weight_bf16.mem"))
    conv_bias    = load_bf16_mem(os.path.join(FP, "ecg_fused_frontend_conv_bias_bf16.mem"))[0]
    cls_w_flat   = load_bf16_mem(os.path.join(FP, "ecg_fused_frontend_classifier_weight_bf16.mem"))
    cls_bias     = load_bf16_mem(os.path.join(FP, "ecg_fused_frontend_classifier_bias_bf16.mem"))

    # Reshape classifier weights: [2, 128]
    assert len(cls_w_flat) == 256, f"Expected 256 FC weights, got {len(cls_w_flat)}"
    cls_weights = [cls_w_flat[:128], cls_w_flat[128:]]

    # ----------------------------------------------------------
    # Load inputs
    # ----------------------------------------------------------
    all_inputs = load_bf16_mem(os.path.join(ID, f"{IP}_bf16.mem"))
    assert len(all_inputs) == args.num_beats * args.signal_len, (
        f"Expected {args.num_beats * args.signal_len} input samples, "
        f"got {len(all_inputs)}"
    )
    beats = [
        all_inputs[i * args.signal_len : (i + 1) * args.signal_len]
        for i in range(args.num_beats)
    ]

    # ----------------------------------------------------------
    # Load golden references
    # ----------------------------------------------------------
    ref_fp32_logits = load_fp32_logits_txt(os.path.join(RD, f"{RP}_fp32_logits.txt"))
    ref_bf16_logits = load_fp32_logits_txt(os.path.join(RD, f"{RP}_bf16_logits.txt"))
    ref_features    = load_fp32_features_txt(os.path.join(RD, f"{RP}_features_fp32.txt"))

    with open(os.path.join(ID, f"{IP}_labels.txt")) as fh:
        labels = [int(x) for x in fh.read().split()]
    with open(os.path.join(ID, f"{IP}_beat_symbols.txt")) as fh:
        symbols = fh.read().split()

    # ----------------------------------------------------------
    # Print header
    # ----------------------------------------------------------
    sep = "=" * 72
    print(sep)
    print("  Pure Python BF16 Hardware Simulation  —  ECG Classifier")
    print("  Pipeline: Conv1d → ReLU → MaxPool1d(2) → Flatten → Linear")
    print(sep)

    print("\n--- Parameters loaded ---")
    print(f"  Conv weights ({len(conv_weights)} taps, BF16):")
    print("    " + "  ".join(f"{w:+.6f}" for w in conv_weights))
    print(f"  Conv bias   : {conv_bias:+.6f}  ({float_to_bf16_hex(conv_bias)})")
    print(f"  FC bias     : [{cls_bias[0]:+.6f}, {cls_bias[1]:+.6f}]")
    print(f"  FC weights  : shape [2, 128], first row first 4 vals = "
          f"{[f'{v:+.4f}' for v in cls_weights[0][:4]]}")

    # ----------------------------------------------------------
    # Run simulation beat by beat
    # ----------------------------------------------------------
    print()
    all_results = []
    for bi, signal in enumerate(beats):
        result = run_pipeline(signal, conv_weights, conv_bias,
                               cls_weights, cls_bias, verbose=args.verbose)
        all_results.append(result)

        logits   = result["logits"]
        pred     = result["pred"]
        true_lbl = labels[bi]
        sym      = symbols[bi]

        r_fp32 = ref_fp32_logits[bi] if bi < len(ref_fp32_logits) else [0.0, 0.0]
        r_bf16 = ref_bf16_logits[bi] if bi < len(ref_bf16_logits) else [0.0, 0.0]
        r_feat = ref_features[bi]    if bi < len(ref_features)    else []

        feat_maxdiff  = max_abs_diff(result["features"], r_feat) if r_feat else float('nan')
        feat_meandiff = mean_abs_diff(result["features"], r_feat) if r_feat else float('nan')

        diff_vs_fp32 = [abs(logits[j] - r_fp32[j]) for j in range(2)]
        diff_vs_bf16 = [abs(logits[j] - r_bf16[j]) for j in range(2)]

        print(f"{'─'*72}")
        print(f"  Beat {bi+1} / {args.num_beats}   symbol={sym}   "
              f"label={true_lbl}   pred={pred}   "
              f"{'✓ correct' if pred == true_lbl else '✗ WRONG'}")
        print()
        print(f"  {'Stage':<20} {'logit[0]':>14}  {'logit[1]':>14}")
        print(f"  {'Sim (BF16 MAC)':<20} {logits[0]:>+14.6f}  {logits[1]:>+14.6f}")
        print(f"  {'Ref BF16 emul':<20} {r_bf16[0]:>+14.6f}  {r_bf16[1]:>+14.6f}")
        print(f"  {'Ref FP32':<20} {r_fp32[0]:>+14.6f}  {r_fp32[1]:>+14.6f}")
        print()
        print(f"  Diff vs Ref-BF16  : [{diff_vs_bf16[0]:.6f}, {diff_vs_bf16[1]:.6f}]")
        print(f"  Diff vs Ref-FP32  : [{diff_vs_fp32[0]:.6f}, {diff_vs_fp32[1]:.6f}]")
        print()
        print(f"  Feature vector vs FP32 reference:")
        print(f"    max |diff|  = {feat_maxdiff:.6f}")
        print(f"    mean|diff|  = {feat_meandiff:.6f}")

        if args.verbose:
            print()
            print(f"  Conv  out (first 16): {[f'{v:.4f}' for v in result['conv'][:16]]}")
            print(f"  ReLU  out (first 16): {[f'{v:.4f}' for v in result['relu'][:16]]}")
            print(f"  Pool  out (first 16): {[f'{v:.4f}' for v in result['pool'][:16]]}")
            print(f"  Feat  ref (first 16): {[f'{v:.4f}' for v in r_feat[:16]]}")

        print()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(sep)
    correct = sum(r["pred"] == labels[i] for i, r in enumerate(all_results))
    print(f"  Simulation accuracy : {correct}/{args.num_beats}  ({100*correct/args.num_beats:.0f}%)")

    all_sim  = [r["logits"] for r in all_results]
    all_feats = [r["features"] for r in all_results]

    flat_sim  = [v for row in all_sim for v in row]
    flat_rbf16 = [v for row in ref_bf16_logits for v in row]
    flat_rfp32 = [v for row in ref_fp32_logits for v in row]
    flat_feat_sim = [v for row in all_feats for v in row]
    flat_feat_ref = [v for row in ref_features for v in row]

    print(f"  Logit max|diff| vs Ref-BF16 : {max_abs_diff(flat_sim, flat_rbf16):.6f}")
    print(f"  Logit max|diff| vs Ref-FP32 : {max_abs_diff(flat_sim, flat_rfp32):.6f}")
    print(f"  Feature max|diff| vs FP32   : {max_abs_diff(flat_feat_sim, flat_feat_ref):.6f}")
    print(f"  Feature mean|diff| vs FP32  : {mean_abs_diff(flat_feat_sim, flat_feat_ref):.6f}")
    print(sep)
    print()
    print("BF16 operation summary")
    print("  bf16_mul(a, b) -> fp32_to_bf16(a * b)      [truncate product]")
    print("  bf16_add(a, b) -> fp32_to_bf16(a + b)      [truncate sum]")
    print("  bf16_mac(acc, a, b) -> bf16_add(acc, bf16_mul(a, b))")
    print()
    print("Conv1d   : 7-tap MAC per output sample (stride=1, pad=3)")
    print("ReLU     : max(0, x) — exact, no rounding")
    print("MaxPool  : max of 2 adjacent values — exact, no rounding")
    print("Linear   : 128-tap MAC per output neuron")
    print(sep)


if __name__ == "__main__":
    main()
