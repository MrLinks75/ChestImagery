from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


PATHOLOGY_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Per-class and macro AUC-ROC. Skips classes with only one label present."""
    per_class = {}
    for i, name in enumerate(PATHOLOGY_NAMES):
        if len(np.unique(y_true[:, i])) > 1:
            per_class[name] = roc_auc_score(y_true[:, i], y_score[:, i])
        else:
            per_class[name] = float("nan")
    valid = [v for v in per_class.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return {"per_class": per_class, "macro_auc": macro}


def compute_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean average precision across classes."""
    aps = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            aps.append(average_precision_score(y_true[:, i], y_score[:, i]))
    return float(np.mean(aps)) if aps else float("nan")
