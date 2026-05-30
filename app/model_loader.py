"""Lazy model loader for Streamlit demo.

Falls back to random weights with a warning if checkpoint is missing,
allowing the UI to be tested without completed training runs.
"""
from __future__ import annotations
import torch
import numpy as np
from pathlib import Path

from src.models.cnn_scratch import CNNScratch
from src.models.densenet_transfer import DenseNetTransfer
from src.models.vit_model import ViTModel
from src.anomaly.vae import VAE

_cache: dict = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load(name: str, model, ckpt_path: str):
    if name not in _cache:
        path = Path(ckpt_path)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=DEVICE))
        else:
            import streamlit as st
            st.warning(f"Checkpoint not found: {ckpt_path} — using random weights")
        model.eval().to(DEVICE)
        _cache[name] = model
    return _cache[name]


def get_classifier(name: str):
    if name == "CNN (scratch)":
        return _load("cnn", CNNScratch(14), "checkpoints/cnn_scratch_best.pth")
    elif name == "DenseNet121":
        return _load("densenet", DenseNetTransfer(14), "checkpoints/densenet121_best.pth")
    elif name == "ViT":
        return _load("vit", ViTModel(14), "checkpoints/vit_base_patch16_224_best.pth")
    raise ValueError(f"Unknown model: {name}")


def get_vae():
    return _load("vae", VAE(128), "checkpoints/vae_best.pth")


def get_vae_threshold() -> float:
    path = Path("checkpoints/vae_threshold.npy")
    if path.exists():
        return float(np.load(path))
    return 0.02  # fallback default
