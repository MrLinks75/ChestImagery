from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np
import mlflow
from tqdm import tqdm

from src.utils.config import Config
from src.utils.metrics import compute_auc, compute_map
from src.utils import mlflow_helpers as mh


class EarlyStopping:
    def __init__(self, patience: int = 10) -> None:
        self.patience = patience
        self.best = -1.0
        self.counter = 0

    def step(self, score: float) -> bool:
        if score > self.best:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(self, model: nn.Module, cfg: Config,
                 train_loader: DataLoader, val_loader: DataLoader,
                 pos_weight: torch.Tensor | None = None,
                 device: str = "cuda") -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        pw = pos_weight.to(device) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        self.early_stop = EarlyStopping(patience=cfg.early_stopping_patience)
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        total_loss = 0.0
        all_labels, all_preds = [], []

        with torch.set_grad_enabled(train):
            for images, labels in tqdm(loader, leave=False):
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(images)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        y_true = np.concatenate(all_labels)
        y_score = np.concatenate(all_preds)
        return avg_loss, y_true, y_score

    def fit(self) -> str:
        best_auc = -1.0
        best_ckpt = ""
        cfg = self.cfg

        for epoch in range(1, cfg.epochs + 1):
            # Progressive unfreezing for transfer models
            if hasattr(self.model, "freeze_backbone") and hasattr(self.model, "unfreeze_backbone"):
                if epoch == 1:
                    self.model.freeze_backbone()
                    # Rebuild optimizer for only trainable params
                    self.optimizer = AdamW(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=cfg.lr, weight_decay=cfg.weight_decay
                    )
                elif epoch == cfg.freeze_epochs + 1:
                    self.model.unfreeze_backbone()
                    self.optimizer = AdamW(
                        self.model.parameters(), lr=cfg.lr * 0.1,
                        weight_decay=cfg.weight_decay
                    )

            if hasattr(self.model, "freeze_backbone") and hasattr(self.model, "unfreeze_backbone"):
                if epoch == cfg.linear_probe_epochs + 1 and cfg.linear_probe_epochs > 0:
                    self.model.unfreeze_backbone()

            train_loss, _, _ = self._run_epoch(self.train_loader, train=True)
            val_loss, y_true, y_score = self._run_epoch(self.val_loader, train=False)
            auc_info = compute_auc(y_true, y_score)
            macro_auc = auc_info["macro_auc"]
            mAP = compute_map(y_true, y_score)

            self.scheduler.step()

            metrics = {
                "train_loss": train_loss, "val_loss": val_loss,
                "val_macro_auc": macro_auc, "val_map": mAP,
            }
            mh.log_epoch_metrics(metrics, step=epoch)
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_AUC={macro_auc:.4f}")

            if macro_auc > best_auc:
                best_auc = macro_auc
                best_ckpt = str(self.checkpoint_dir / f"{cfg.model}_best.pth")
                torch.save(self.model.state_dict(), best_ckpt)
                mlflow.log_metric("best_val_auc", best_auc, step=epoch)

            if self.early_stop.step(macro_auc):
                print(f"Early stopping at epoch {epoch}")
                break

        return best_ckpt
