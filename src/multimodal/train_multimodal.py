"""Train and compare the three multimodal models on OpenI dataset."""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import mlflow
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.metrics import compute_auc
from src.utils import mlflow_helpers as mh
from src.data.openi_dataset import get_openi_loaders
from src.multimodal.fusion_models import ImageOnlyModel, TextOnlyModel, LateFusionModel


def run_experiment(model_name: str, model: nn.Module, cfg,
                   train_loader, val_loader, test_loader, device: str) -> None:
    ckpt_path = Path(f"checkpoints/multimodal_{model_name}_best.pth")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -1.0

    with mlflow.start_run(run_name=model_name, nested=True):
        mh.log_config(cfg)
        mlflow.log_param("variant", model_name)

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            for batch in tqdm(train_loader, leave=False):
                imgs, input_ids, attn_mask, labels = [b.to(device) for b in batch]
                logits = model(imgs, input_ids, attn_mask)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation AUC
            model.eval()
            all_labels, all_preds = [], []
            with torch.no_grad():
                for batch in val_loader:
                    imgs, input_ids, attn_mask, labels = [b.to(device) for b in batch]
                    logits = model(imgs, input_ids, attn_mask)
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())

            y_true = np.concatenate(all_labels)
            y_score = np.concatenate(all_preds)
            auc = compute_auc(y_true, y_score)["macro_auc"]
            mlflow.log_metric("val_macro_auc", auc, step=epoch)

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), ckpt_path)

        mlflow.log_metric("best_val_auc", best_auc)
        mh.log_model_checkpoint(ckpt_path)
        print(f"{model_name}: best val AUC = {best_auc:.4f}")


def main(cfg_path: str, data_dir: str, image_ckpt: str | None = None) -> None:
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = get_openi_loaders(
        data_dir, img_size=cfg.img_size, batch_size=cfg.batch_size,
        text_model=cfg.text_model
    )

    models = {
        "image_only": ImageOnlyModel(cfg.num_classes, image_ckpt),
        "text_only": TextOnlyModel(cfg.num_classes, cfg.text_model, cfg.freeze_text_encoder),
        "late_fusion": LateFusionModel(
            cfg.num_classes, cfg.text_model, cfg.freeze_text_encoder,
            cfg.fusion_hidden_dim, image_ckpt
        ),
    }

    mh.init_mlflow("multimodal_comparison")
    with mlflow.start_run(run_name="multimodal_comparison"):
        for name, model in models.items():
            run_experiment(name, model.to(device), cfg,
                           train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument("--data_dir", required=True, help="Path to data/openi/")
    parser.add_argument("--image_ckpt", default=None,
                        help="DenseNet checkpoint to initialize image encoder")
    args = parser.parse_args()
    main(args.config, args.data_dir, args.image_ckpt)
