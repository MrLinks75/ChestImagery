"""Vision Transformer via timm.

ViT splits the image into fixed-size patches (16x16), projects them to
embeddings, and applies multi-head self-attention across all patches.
This captures long-range dependencies that local convolutions miss —
relevant for bilateral pathologies or diffuse infiltrates.

Trade-off vs CNN: ViT needs more data / pretraining; we use ImageNet-21k
pretrained weights and fine-tune in two stages.
"""
import timm
import torch
import torch.nn as nn


class ViTModel(nn.Module):
    def __init__(self, num_classes: int = 14,
                 backbone: str = "vit_base_patch16_224") -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, num_classes=0  # remove head
        )
        embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
