"""Entry point for VAE anomaly detection training.

Usage: python scripts/train_vae.py --config configs/vae.yaml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anomaly.train_vae import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vae.yaml")
    args = parser.parse_args()
    main(args.config)
