from __future__ import annotations

import torch
from torch import nn


class ECGFeatureExtractor(nn.Module):
    """Single Conv1d kernel -> BatchNorm -> ReLU -> MaxPool -> Flatten."""

    def __init__(self, input_channels: int = 1, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(input_channels, 1, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.flatten(start_dim=1)


class ECGNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        signal_length: int = 256,
        num_classes: int = 2,
        kernel_size: int = 7,
        hidden_channels: list[int] | None = None,
        mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = ECGFeatureExtractor(
            input_channels=input_channels,
            kernel_size=kernel_size,
        )
        self.feature_dim = self._infer_feature_dim(input_channels, signal_length)
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
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def build_model_from_checkpoint(checkpoint: dict) -> ECGNet:
    config = checkpoint.get("config", {})
    model = ECGNet(
        input_channels=checkpoint["input_channels"],
        signal_length=checkpoint["signal_length"],
        num_classes=checkpoint["num_classes"],
        kernel_size=config.get("kernel_size", 7),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def fuse_conv_bn_1d(
    conv: nn.Conv1d,
    bn: nn.BatchNorm1d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a Conv1d + BatchNorm1d pair into equivalent conv parameters."""

    if conv.out_channels != bn.num_features:
        raise ValueError(
            "Conv out_channels must match BatchNorm num_features for fusion."
        )

    conv_weight = conv.weight.detach()
    if conv.bias is None:
        conv_bias = torch.zeros(conv.out_channels, device=conv_weight.device, dtype=conv_weight.dtype)
    else:
        conv_bias = conv.bias.detach()

    bn_weight = bn.weight.detach()
    bn_bias = bn.bias.detach()
    running_mean = bn.running_mean.detach()
    running_var = bn.running_var.detach()
    eps = bn.eps

    inv_std = torch.rsqrt(running_var + eps)
    scale = bn_weight * inv_std

    fused_weight = conv_weight * scale.view(-1, 1, 1)
    fused_bias = bn_bias + (conv_bias - running_mean) * scale
    return fused_weight, fused_bias


def fuse_feature_extractor(
    feature_extractor: ECGFeatureExtractor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return equivalent Conv1d parameters for the feature extractor's Conv+BN block."""

    return fuse_conv_bn_1d(feature_extractor.conv, feature_extractor.bn)
