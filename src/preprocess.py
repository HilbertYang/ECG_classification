from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitArrays:
    signals: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class PreparedSplits:
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays


def ensure_signal_shape(signals: np.ndarray) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim == 2:
        signals = signals[:, np.newaxis, :]
    if signals.ndim != 3:
        raise ValueError(
            "Signals must have shape (num_samples, signal_length) or "
            "(num_samples, channels, signal_length)."
        )
    return signals


def resize_signals(signals: np.ndarray, target_length: int) -> np.ndarray:
    current_length = signals.shape[-1]
    if current_length == target_length:
        return signals
    if current_length > target_length:
        start = (current_length - target_length) // 2
        end = start + target_length
        return signals[..., start:end]

    padded = np.zeros(
        (signals.shape[0], signals.shape[1], target_length),
        dtype=signals.dtype,
    )
    padded[..., :current_length] = signals
    return padded


def normalize_per_sample(signals: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = signals.mean(axis=-1, keepdims=True)
    std = signals.std(axis=-1, keepdims=True)
    return (signals - mean) / np.maximum(std, eps)


def to_binary_labels(labels: np.ndarray, normal_label: int = 0) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    unique_values = set(np.unique(labels).tolist())
    if unique_values.issubset({0, 1}):
        return labels.astype(np.int64)
    return np.where(labels == normal_label, 0, 1).astype(np.int64)


def _split_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    stratify = labels if len(np.unique(labels)) > 1 else None
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    return train_idx, test_idx


def _empty_like(signals: np.ndarray, labels: np.ndarray) -> SplitArrays:
    empty_signals = np.empty((0, signals.shape[1], signals.shape[2]), dtype=signals.dtype)
    empty_labels = np.empty((0,), dtype=labels.dtype)
    return SplitArrays(signals=empty_signals, labels=empty_labels)


def prepare_splits(
    signals: np.ndarray,
    labels: np.ndarray,
    target_length: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    normal_label: int = 0,
    seed: int = 42,
) -> PreparedSplits:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1.")

    signals = ensure_signal_shape(signals)
    signals = resize_signals(signals, target_length)
    signals = normalize_per_sample(signals)
    labels = to_binary_labels(labels, normal_label=normal_label)

    indices = np.arange(len(labels))
    holdout_ratio = val_ratio + test_ratio

    if holdout_ratio == 0:
        train = SplitArrays(signals=signals, labels=labels)
        empty = _empty_like(signals, labels)
        return PreparedSplits(train=train, val=empty, test=empty)

    train_idx, holdout_idx = _split_indices(indices, labels, holdout_ratio, seed)
    train = SplitArrays(signals=signals[train_idx], labels=labels[train_idx])

    if val_ratio == 0:
        val = _empty_like(signals, labels)
        test = SplitArrays(signals=signals[holdout_idx], labels=labels[holdout_idx])
        return PreparedSplits(train=train, val=val, test=test)

    if test_ratio == 0:
        val = SplitArrays(signals=signals[holdout_idx], labels=labels[holdout_idx])
        test = _empty_like(signals, labels)
        return PreparedSplits(train=train, val=val, test=test)

    relative_test_ratio = test_ratio / holdout_ratio
    val_idx, test_idx = _split_indices(
        holdout_idx,
        labels[holdout_idx],
        relative_test_ratio,
        seed,
    )
    val = SplitArrays(signals=signals[val_idx], labels=labels[val_idx])
    test = SplitArrays(signals=signals[test_idx], labels=labels[test_idx])
    return PreparedSplits(train=train, val=val, test=test)
