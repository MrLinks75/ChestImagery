"""Three multimodal models for ablation comparison.

Fusion strategy: Late fusion (concatenation of image and text embeddings).
Justified because:
  - Simplest to implement and interpret
  - Each modality is first encoded independently, allowing clean ablation
  - Avoids alignment assumptions between pixel-level and token-level features
  - Appropriate for proof-of-concept on a small dataset (OpenI ~3600 pairs)
"""
import torch
import torch.nn as nn
from src.multimodal.text_encoder import BertEncoder
from src.models.densenet_transfer import DenseNetTransfer


class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes: int = 14,
                 image_ckpt: str | None = None) -> None:
        super().__init__()
        self.image_encoder = DenseNetTransfer(num_classes=num_classes)
        if image_ckpt:
            state = torch.load(image_ckpt, map_location="cpu")
            self.image_encoder.load_state_dict(state, strict=False)
        # Replace head for this dataset
        self.image_encoder.classifier = nn.Linear(1024, num_classes)

    def forward(self, images: torch.Tensor, input_ids=None,
                attention_mask=None) -> torch.Tensor:
        return self.image_encoder(images)


class TextOnlyModel(nn.Module):
    def __init__(self, num_classes: int = 14,
                 text_model: str = "bert-base-uncased",
                 freeze_text: bool = True) -> None:
        super().__init__()
        self.text_encoder = BertEncoder(text_model, freeze=freeze_text)
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_encoder(input_ids, attention_mask)
        return self.head(text_emb)


class LateFusionModel(nn.Module):
    def __init__(self, num_classes: int = 14,
                 text_model: str = "bert-base-uncased",
                 freeze_text: bool = True,
                 fusion_hidden_dim: int = 512,
                 image_ckpt: str | None = None) -> None:
        super().__init__()
        backbone = DenseNetTransfer(num_classes=0)
        if image_ckpt:
            state = torch.load(image_ckpt, map_location="cpu")
            backbone.load_state_dict(state, strict=False)
        self.image_encoder = backbone  # use get_embedding()

        self.text_encoder = BertEncoder(text_model, freeze=freeze_text)

        self.fusion_head = nn.Sequential(
            nn.Linear(1024 + 768, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        img_emb = self.image_encoder.get_embedding(images)
        txt_emb = self.text_encoder(input_ids, attention_mask)
        fused = torch.cat([img_emb, txt_emb], dim=1)
        return self.fusion_head(fused)
