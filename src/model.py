from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ECGFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: Sequence[int] = (16, 32, 64),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channel_sizes = [input_channels, *hidden_channels]

        for in_channels, out_channels in zip(channel_sizes, channel_sizes[1:]):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.pool(x)
        return x.flatten(start_dim=1)


class ECGNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        signal_length: int = 256,
        num_classes: int = 2,
        hidden_channels: Sequence[int] = (16, 32, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_extractor = ECGFeatureExtractor(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
        )
        self.feature_dim = self._infer_feature_dim(input_channels, signal_length)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _infer_feature_dim(self, input_channels: int, signal_length: int) -> int:
        with torch.no_grad():
            sample = torch.zeros(1, input_channels, signal_length)
            features = self.feature_extractor(sample)
        return int(features.shape[1])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features
        return logits


def build_model_from_checkpoint(checkpoint: dict) -> ECGNet:
    config = checkpoint.get("config", {})
    model = ECGNet(
        input_channels=checkpoint["input_channels"],
        signal_length=checkpoint["signal_length"],
        num_classes=checkpoint["num_classes"],
        hidden_channels=config.get("hidden_channels", [16, 32, 64]),
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model
