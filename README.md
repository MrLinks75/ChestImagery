# ChestImagery — Radiological Triage System

Multi-label chest X-ray classification (14 pathologies), anomaly detection via VAE, multimodal image+text fusion, and an interactive Streamlit demo. All experiments tracked with MLflow.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (fast package manager)
- NVIDIA GPU with CUDA 12.1 (recommended — CPU works but is very slow for ViT)

---

## 1. Clone & Environment Setup

```bash
git clone https://github.com/MrLinks75/ChestImagery.git
cd ChestImagery

# Create virtual environment with uv
uv venv .venv --python 3.11
```

**Activate the environment:**
```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## 2. Install Dependencies

```bash
# CPU-only (no GPU)
uv pip install -r requirements.txt

# CUDA 12.1 (recommended — replaces torch with GPU build)
uv pip install -r requirements.txt
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --reinstall
```

**Verify CUDA is detected:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## 3. Data

**ChestMNIST** (primary — 14-class multi-label, ~1.4 GB) downloads automatically on first training run via `medmnist`. No manual steps needed.

**OpenI** (optional — multimodal image + radiology reports):
1. Download from https://openi.nlm.nih.gov/
2. Place files as:
```
data/openi/
├── images/   # *.png chest X-rays
└── reports/  # *.xml radiology reports
```

---

## 4. Training

All models log to MLflow automatically. Run in order:

```bash
# CNN from scratch (baseline, ~60 min on GPU)
python scripts/train_classifier.py --config configs/cnn_scratch.yaml

# DenseNet121 — transfer learning / CheXNet architecture (~60–80 min)
python scripts/train_classifier.py --config configs/densenet.yaml

# Vision Transformer — ViT-B/16 (~60–90 min, GPU required)
python scripts/train_classifier.py --config configs/vit.yaml

# VAE anomaly detection (~45–60 min)
python scripts/train_vae.py --config configs/vae.yaml

# Multimodal fusion — image + report text (requires OpenI data)
python scripts/train_multimodal.py \
    --config configs/multimodal.yaml \
    --data_dir data/openi \
    --image_ckpt checkpoints/densenet121_best.pth
```

Checkpoints are saved to `checkpoints/<model>_best.pth`.

---

## 5. MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 — view all runs, compare metrics, download artifacts.

---

## 6. Streamlit Demo

```bash
streamlit run app/app.py
```

Upload a chest X-ray to get:
- Per-pathology probability bar chart (14 classes)
- VAE anomaly score with color-coded indicator
- Optional: paste a radiology report for multimodal fusion comparison

> Note: train at least one classifier and the VAE before running the demo, or it will load with random weights and warn you.

---

## 7. Smoke Test

Verifies all 5 model types pass a forward pass without any training data:

```bash
python scripts/smoke_test.py
```

---

## Project Structure

```
ChestImagery/
├── app/                    # Streamlit demo
│   ├── app.py
│   └── model_loader.py
├── configs/                # YAML hyperparameter configs
│   ├── cnn_scratch.yaml
│   ├── densenet.yaml
│   ├── vit.yaml
│   ├── vae.yaml
│   └── multimodal.yaml
├── scripts/                # Training entry points
│   ├── train_classifier.py
│   ├── train_vae.py
│   ├── smoke_test.py
│   └── verify_mlflow_consistency.py
├── src/
│   ├── data/               # Dataset classes (ChestMNIST, OpenI)
│   ├── models/             # CNN, DenseNet121, ViT
│   ├── training/           # Shared Trainer, evaluation, metrics
│   ├── anomaly/            # VAE architecture + training
│   ├── multimodal/         # BERT encoder, fusion models
│   └── utils/              # Config, seed, MLflow helpers, visualization
├── requirements.txt
└── README.md
```

---

## Hardware Notes

| Model | img_size | batch_size | Est. VRAM | Est. time (RTX 3070) |
|---|---|---|---|---|
| CNN scratch | 128 | 128 | ~2 GB | ~60 min |
| DenseNet121 | 128 | 64 | ~4 GB | ~70 min |
| ViT-B/16 | 224 | 32 | ~6 GB | ~80 min |
| VAE | 128 | 128 | ~2 GB | ~45 min |
| Late Fusion | 128 | 32 | ~5 GB | ~40 min |

---

## Reproducibility

- Fixed seed `42` across all scripts (configurable in each YAML)
- `torch.backends.cudnn.deterministic = True` enabled globally
- medmnist provides patient-safe train/val/test splits (no leakage)
