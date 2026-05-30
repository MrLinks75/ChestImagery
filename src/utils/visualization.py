from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve
from src.utils.metrics import PATHOLOGY_NAMES


def plot_roc_curves(y_true: np.ndarray, y_score: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(PATHOLOGY_NAMES):
        if len(np.unique(y_true[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
            ax.plot(fpr, tpr, label=name, linewidth=1)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves per Pathology")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reconstruction_grid(originals: np.ndarray, reconstructions: np.ndarray,
                              scores: np.ndarray, save_path: Path, n: int = 8) -> None:
    """Side-by-side original vs reconstruction for top-n anomaly scores."""
    idx = np.argsort(scores)[-n:]
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    for col, i in enumerate(idx):
        axes[0, col].imshow(originals[i].squeeze(), cmap="gray")
        axes[0, col].set_title(f"score={scores[i]:.2f}", fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(reconstructions[i].squeeze(), cmap="gray")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=8)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
