"""Custom CNN trained from scratch.

Architecture justification (course concepts):
- Conv layers: learn local spatial features (edges, textures, patterns)
- BatchNorm: stabilizes training, allows higher lr
- ReLU: non-linearity without vanishing gradient
- MaxPool: spatial downsampling, translation invariance
- GlobalAvgPool: replaces large FC, reduces overfitting
- Sigmoid per class: independent binary prediction for each pathology
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNScratch(nn.Module):
    def __init__(self, num_classes: int = 14) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 128 -> 64
            ConvBlock(32, 64),   # 64 -> 32
            ConvBlock(64, 128),  # 32 -> 16
            ConvBlock(128, 256), # 16 -> 8
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)
