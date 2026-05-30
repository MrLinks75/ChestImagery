"""Frozen BERT text encoder for multimodal fusion.

We freeze BERT weights to avoid expensive fine-tuning on the small OpenI
dataset. Only the projection head is trained. Using frozen BERT as a
feature extractor is justified as a proof-of-concept for image+text fusion.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class BertEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased",
                 freeze: bool = True) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns [CLS] token embedding, shape (B, 768)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # [CLS]
