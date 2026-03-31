from __future__ import annotations

import numpy as np


def float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    fp32_bits = values.view(np.uint32)

    lsb = (fp32_bits >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    bf16_bits = ((fp32_bits + rounding_bias) >> 16).astype(np.uint16)

    nan_mask = ((fp32_bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & (
        (fp32_bits & np.uint32(0x007FFFFF)) != np.uint32(0)
    )
    if np.any(nan_mask):
        bf16_bits[nan_mask] = ((fp32_bits[nan_mask] >> 16) | np.uint32(0x0040)).astype(np.uint16)

    return bf16_bits


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint16)
    fp32_bits = bits.astype(np.uint32) << np.uint32(16)
    return fp32_bits.view(np.float32)


def round_to_bf16(values: np.ndarray) -> np.ndarray:
    return bf16_bits_to_float32(float32_to_bf16_bits(values))


def flatten_weight(weight: np.ndarray, layout: str) -> np.ndarray:
    if layout == "row-major":
        return np.asarray(weight, dtype=np.float32).reshape(-1)
    if layout == "col-major":
        return np.asarray(weight, dtype=np.float32).transpose(1, 0).reshape(-1)
    raise ValueError(f"Unsupported layout: {layout}")


def to_hex_lines(values: np.ndarray, word_bits: int = 16) -> list[str]:
    width = (word_bits + 3) // 4
    mask = (1 << word_bits) - 1
    return [format(int(value) & mask, f"0{width}X") for value in np.asarray(values).reshape(-1)]


def bf16_hex_lines(values: np.ndarray) -> list[str]:
    return to_hex_lines(float32_to_bf16_bits(values), word_bits=16)


def bf16_linear(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    accumulation: str = "fp32",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError("Features must have shape (batch, input_dim).")
    if weight.ndim != 2:
        raise ValueError("Weight must have shape (output_dim, input_dim).")
    if bias.ndim != 1:
        raise ValueError("Bias must have shape (output_dim,).")
    if features.shape[1] != weight.shape[1]:
        raise ValueError("Feature dimension must match the classifier input dimension.")
    if weight.shape[0] != bias.shape[0]:
        raise ValueError("Bias length must match the classifier output dimension.")

    features_bf16 = round_to_bf16(features)
    weight_bf16 = round_to_bf16(weight)
    bias_bf16 = round_to_bf16(bias)

    if accumulation == "fp32":
        logits = features_bf16 @ weight_bf16.transpose(1, 0) + bias_bf16
        return logits.astype(np.float32), features_bf16, weight_bf16, bias_bf16, logits.astype(np.float32)

    if accumulation != "bf16":
        raise ValueError("accumulation must be 'fp32' or 'bf16'.")

    batch_size, input_dim = features_bf16.shape
    output_dim = weight_bf16.shape[0]
    logits = np.empty((batch_size, output_dim), dtype=np.float32)

    for batch_index in range(batch_size):
        for output_index in range(output_dim):
            acc = bias_bf16[output_index]
            for input_index in range(input_dim):
                product = round_to_bf16(
                    np.array(
                        [features_bf16[batch_index, input_index] * weight_bf16[output_index, input_index]],
                        dtype=np.float32,
                    )
                )[0]
                acc = round_to_bf16(np.array([acc + product], dtype=np.float32))[0]
            logits[batch_index, output_index] = acc

    return logits, features_bf16, weight_bf16, bias_bf16, logits
