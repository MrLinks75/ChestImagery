"""Train VAE on normal (label-free) ChestMNIST samples."""
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import mlflow
from pathlib import Path
from tqdm import tqdm

from src.anomaly.vae import VAE, vae_loss
from src.data.chestmnist_dataset import get_normal_indices
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils import mlflow_helpers as mh
from src.utils.visualization import plot_reconstruction_grid
import medmnist
from medmnist import ChestMNIST


def get_grayscale_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1] grayscale
    ])


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = get_grayscale_transform(cfg.img_size)
    train_full = ChestMNIST(split="train", transform=tf, download=True,
                             size=cfg.img_size, as_rgb=False)
    val_full = ChestMNIST(split="val", transform=tf, download=True,
                           size=cfg.img_size, as_rgb=False)

    train_idx = get_normal_indices(train_full)
    val_idx = get_normal_indices(val_full)
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    model = VAE(latent_dim=cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    mh.init_mlflow("anomaly_detection")
    with mlflow.start_run(run_name="vae"):
        mh.log_config(cfg)
        best_val_loss = float("inf")
        ckpt_path = Path("checkpoints/vae_best.pth")
        ckpt_path.parent.mkdir(exist_ok=True)

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_loss = 0.0
            for (imgs, _) in tqdm(train_loader, leave=False):
                imgs = imgs.to(device)
                recon, mu, logvar = model(imgs)
                loss = vae_loss(recon, imgs, mu, logvar, beta=cfg.beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (imgs, _) in val_loader:
                    imgs = imgs.to(device)
                    recon, mu, logvar = model(imgs)
                    val_loss += vae_loss(recon, imgs, mu, logvar, beta=cfg.beta).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            mh.log_epoch_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_path)

        # Compute threshold on val set anomaly scores
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        scores = []
        originals, reconstructions = [], []
        with torch.no_grad():
            for (imgs, _) in val_loader:
                imgs = imgs.to(device)
                s = model.anomaly_score(imgs)
                scores.append(s.cpu().numpy())
                recon, _, _ = model(imgs)
                originals.append(imgs.cpu().numpy())
                reconstructions.append(recon.cpu().numpy())

        scores = np.concatenate(scores)
        threshold = float(np.percentile(scores, cfg.anomaly_threshold_percentile))
        mlflow.log_metric("anomaly_threshold", threshold)
        np.save("checkpoints/vae_threshold.npy", threshold)

        orig_arr = np.concatenate(originals)
        recon_arr = np.concatenate(reconstructions)
        recon_path = Path("artifacts/vae_reconstructions.png")
        recon_path.parent.mkdir(exist_ok=True)
        plot_reconstruction_grid(orig_arr, recon_arr, scores, recon_path)
        mh.log_artifact(recon_path)
        mh.log_model_checkpoint(ckpt_path)
        print(f"Anomaly threshold (p{cfg.anomaly_threshold_percentile}): {threshold:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vae.yaml")
    args = parser.parse_args()
    main(args.config)
