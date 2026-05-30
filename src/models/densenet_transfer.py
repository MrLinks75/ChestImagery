"""DenseNet121 with transfer learning.

Chosen for its relevance to medical imaging: CheXNet (Rajpurkar et al. 2017)
used exactly this architecture on chest X-rays. Dense connections reuse
feature maps from all previous layers, reducing parameters while maintaining
gradient flow.

Fine-tuning strategy:
- Epochs 1..freeze_epochs: backbone frozen, only head trained
- Epochs freeze_epochs+1..: full network unfrozen
"""
import torch
import torch.nn as nn
import torchvision.models as models


class DenseNetTransfer(nn.Module):
    def __init__(self, num_classes: int = 14) -> None:
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """1024-dim embedding for multimodal fusion."""
        x = self.features(x)
        x = torch.relu(x)
        return self.gap(x).flatten(1)
