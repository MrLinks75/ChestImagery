"""Entry point for training CNN scratch, DenseNet, or ViT classifiers.

Usage:
  python scripts/train_classifier.py --config configs/cnn_scratch.yaml
  python scripts/train_classifier.py --config configs/densenet.yaml
  python scripts/train_classifier.py --config configs/vit.yaml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import mlflow
import torch
from pathlib import Path

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils import mlflow_helpers as mh
from src.data.chestmnist_dataset import get_chestmnist_loaders, compute_pos_weights
from src.training.trainer import Trainer
from src.training.evaluate import evaluate


MODEL_REGISTRY = {
    "cnn_scratch": ("src.models.cnn_scratch", "CNNScratch"),
    "densenet121": ("src.models.densenet_transfer", "DenseNetTransfer"),
    "vit_base_patch16_224": ("src.models.vit_model", "ViTModel"),
}


def build_model(cfg):
    import importlib
    module_path, class_name = MODEL_REGISTRY[cfg.model]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(num_classes=cfg.num_classes)


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {cfg.model} on {device}")

    train_loader, val_loader, test_loader = get_chestmnist_loaders(
        img_size=cfg.img_size, batch_size=cfg.batch_size
    )
    pos_weight = compute_pos_weights(train_loader, cfg.num_classes)
    model = build_model(cfg)

    mh.init_mlflow("supervised_classification")
    with mlflow.start_run(run_name=cfg.model):
        mh.log_config(cfg)
        trainer = Trainer(model, cfg, train_loader, val_loader, pos_weight, device)
        best_ckpt = trainer.fit()

        # Reload best weights for test evaluation
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        results = evaluate(model, test_loader, device,
                           artifact_dir=Path(f"artifacts/{cfg.model}"))
        mh.log_model_checkpoint(best_ckpt)
        print(f"Done. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
