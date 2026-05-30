from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow
from tqdm import tqdm

from src.utils.metrics import compute_auc, compute_map, PATHOLOGY_NAMES
from src.utils.visualization import plot_roc_curves
from src.utils import mlflow_helpers as mh


def evaluate(model: nn.Module, test_loader: DataLoader,
             device: str = "cuda", artifact_dir: Path = Path("artifacts")) -> dict:
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            all_labels.append(labels.float().cpu().numpy())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_preds)

    auc_info = compute_auc(y_true, y_score)
    mAP = compute_map(y_true, y_score)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    roc_path = artifact_dir / "roc_curves.png"
    plot_roc_curves(y_true, y_score, roc_path)
    mh.log_artifact(roc_path)

    results = {"macro_auc": auc_info["macro_auc"], "mAP": mAP, **auc_info["per_class"]}
    mlflow.log_metrics({k: v for k, v in results.items() if not np.isnan(v)})

    print(f"\nTest macro AUC: {auc_info['macro_auc']:.4f} | mAP: {mAP:.4f}")
    for name, auc in auc_info["per_class"].items():
        print(f"  {name:<25} AUC={auc:.4f}")

    return results
