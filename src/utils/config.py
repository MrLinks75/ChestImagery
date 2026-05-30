from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class Config:
    model: str = "cnn_scratch"
    seed: int = 42
    img_size: int = 128
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    augmentation: bool = True
    num_classes: int = 14
    # model-specific (optional)
    freeze_epochs: int = 0
    linear_probe_epochs: int = 0
    latent_dim: int = 128
    beta: float = 1.0
    anomaly_threshold_percentile: int = 95
    image_embed_dim: int = 1024
    text_embed_dim: int = 768
    fusion_hidden_dim: int = 512
    max_text_length: int = 256
    text_model: str = "bert-base-uncased"
    freeze_text_encoder: bool = True


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    cfg = Config()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
