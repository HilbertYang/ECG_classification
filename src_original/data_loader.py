from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .preprocess import PreparedSplits, SplitArrays, prepare_splits


@dataclass(frozen=True)
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    splits: PreparedSplits
    input_channels: int
    signal_length: int
    num_classes: int
    source_name: str


def load_npz_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as bundle:
        signals_key = "signals" if "signals" in bundle.files else "x"
        labels_key = "labels" if "labels" in bundle.files else "y"
        if signals_key not in bundle.files or labels_key not in bundle.files:
            raise KeyError(
                "Dataset .npz file must contain either signals/labels or x/y arrays."
            )
        signals = np.asarray(bundle[signals_key], dtype=np.float32)
        labels = np.asarray(bundle[labels_key]).reshape(-1)

    if len(signals) != len(labels):
        raise ValueError("Signals and labels must have the same number of samples.")
    return signals, labels


def generate_demo_dataset(
    num_samples: int = 600,
    signal_length: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0.0, 1.0, signal_length, dtype=np.float32)
    signals = np.zeros((num_samples, signal_length), dtype=np.float32)
    labels = np.zeros((num_samples,), dtype=np.int64)

    base_centers = np.array([0.18, 0.42, 0.66, 0.88], dtype=np.float32)

    for index in range(num_samples):
        label = int(rng.integers(0, 2))
        labels[index] = label

        heart_rate = rng.uniform(1.0, 1.7)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal = 0.07 * np.sin(2.0 * np.pi * heart_rate * time_axis + phase)
        signal += 0.03 * np.sin(2.0 * np.pi * (heart_rate * 3.0) * time_axis)
        signal += rng.normal(0.0, 0.025, size=signal_length)

        beat_centers = base_centers.copy()
        if label == 1:
            beat_centers += rng.normal(0.0, 0.025, size=beat_centers.shape[0])
            signal += 0.08 * np.sin(2.0 * np.pi * 9.0 * time_axis)

        for center in beat_centers:
            width = 0.012 if label == 0 else rng.uniform(0.008, 0.02)
            amplitude = 0.85 if label == 0 else rng.uniform(0.3, 1.1)
            signal += amplitude * np.exp(-0.5 * ((time_axis - center) / width) ** 2)

        if label == 1:
            dip_center = rng.uniform(0.25, 0.75)
            signal -= 0.55 * np.exp(-0.5 * ((time_axis - dip_center) / 0.018) ** 2)

        signals[index] = signal.astype(np.float32)

    return signals, labels


def _to_dataset(split: SplitArrays) -> TensorDataset:
    signals = torch.from_numpy(split.signals).float()
    labels = torch.from_numpy(split.labels).long()
    return TensorDataset(signals, labels)


def create_dataloaders(
    dataset_path: str | Path | None,
    use_demo: bool,
    demo_samples: int,
    signal_length: int,
    batch_size: int,
    val_ratio: float,
    test_ratio: float,
    normal_label: int,
    seed: int,
) -> LoaderBundle:
    use_demo = use_demo or dataset_path is None
    if use_demo:
        signals, labels = generate_demo_dataset(
            num_samples=demo_samples,
            signal_length=signal_length,
            seed=seed,
        )
        source_name = "demo"
    else:
        signals, labels = load_npz_dataset(dataset_path)
        source_name = Path(dataset_path).stem

    splits = prepare_splits(
        signals=signals,
        labels=labels,
        target_length=signal_length,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normal_label=normal_label,
        seed=seed,
    )

    train_loader = DataLoader(_to_dataset(splits.train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_to_dataset(splits.val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_to_dataset(splits.test), batch_size=batch_size, shuffle=False)

    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        splits=splits,
        input_channels=splits.train.signals.shape[1],
        signal_length=splits.train.signals.shape[-1],
        num_classes=2,
        source_name=source_name,
    )
