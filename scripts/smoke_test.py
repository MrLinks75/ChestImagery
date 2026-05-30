"""Forward-pass smoke test for all 5 model types.

Run: python scripts/smoke_test.py
All assertions must pass before any training run.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

B, C, H = 4, 3, 128
x_rgb = torch.randn(B, C, H, H)
x_gray = torch.randn(B, 1, H, H)
x_rgb_224 = torch.randn(B, C, 224, 224)
dummy_ids = torch.randint(0, 1000, (B, 64))
dummy_mask = torch.ones(B, 64, dtype=torch.long)


def check(name: str, logits: torch.Tensor, expected_shape: tuple) -> None:
    assert logits.shape == expected_shape, f"{name}: got {logits.shape}, want {expected_shape}"
    assert not torch.isnan(logits).any(), f"{name}: NaN in output"
    print(f"  PASS  {name}")


print("Running smoke tests...")

from src.models.cnn_scratch import CNNScratch
m = CNNScratch(14)
check("CNNScratch", m(x_rgb), (B, 14))

from src.models.densenet_transfer import DenseNetTransfer
m = DenseNetTransfer(14)
check("DenseNetTransfer", m(x_rgb), (B, 14))
emb = m.get_embedding(x_rgb)
assert emb.shape == (B, 1024), f"DenseNet embedding shape wrong: {emb.shape}"
print("  PASS  DenseNetTransfer.get_embedding")

from src.models.vit_model import ViTModel
m = ViTModel(14)
check("ViTModel", m(x_rgb_224), (B, 14))

from src.anomaly.vae import VAE
vae = VAE(latent_dim=128)
recon, mu, logvar = vae(x_gray)
assert recon.shape == x_gray.shape, f"VAE recon shape wrong: {recon.shape}"
assert not torch.isnan(recon).any(), "VAE: NaN in reconstruction"
scores = vae.anomaly_score(x_gray)
assert scores.shape == (B,), f"VAE anomaly score shape wrong: {scores.shape}"
print("  PASS  VAE (recon + anomaly_score)")

from src.multimodal.fusion_models import LateFusionModel
m = LateFusionModel(14, freeze_text=True)
check("LateFusionModel", m(x_rgb, dummy_ids, dummy_mask), (B, 14))

print("\nAll smoke tests passed.")
