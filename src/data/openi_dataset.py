"""OpenI dataset loader (images + radiology reports from XML).

Expected directory structure after download:
  data/openi/
    images/   -> *.png
    reports/  -> *.xml  (Indiana University Chest X-ray collection)
"""
from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import AutoTokenizer

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _parse_report(xml_path: Path) -> str:
    """Extract findings + impression text from OpenI XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        texts = []
        for section in root.iter("AbstractText"):
            label = section.get("Label", "")
            if label.upper() in ("FINDINGS", "IMPRESSION") and section.text:
                texts.append(section.text.strip())
        return " ".join(texts) if texts else ""
    except Exception:
        return ""


class OpenIDataset(Dataset):
    def __init__(self, data_dir: str | Path, img_size: int = 128,
                 text_model: str = "bert-base-uncased", max_length: int = 256,
                 augment: bool = False):
        self.data_dir = Path(data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.max_length = max_length

        tf_list = [transforms.Resize((img_size, img_size))]
        if augment:
            tf_list += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        tf_list += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.transform = transforms.Compose(tf_list)

        img_dir = self.data_dir / "images"
        report_dir = self.data_dir / "reports"
        self.samples: list[tuple[Path, str]] = []
        for xml_path in sorted(report_dir.glob("*.xml")):
            uid = xml_path.stem
            img_path = img_dir / f"{uid}.png"
            if img_path.exists():
                report_text = _parse_report(xml_path)
                self.samples.append((img_path, report_text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # OpenI does not have standard multi-label ground truth; return dummy labels
        # Replace with actual label parsing if annotation file is available
        labels = torch.zeros(14, dtype=torch.float)
        return image, input_ids, attention_mask, labels


def get_openi_loaders(data_dir: str, img_size: int = 128, batch_size: int = 32,
                       text_model: str = "bert-base-uncased",
                       num_workers: int = 4) -> tuple[DataLoader, DataLoader, DataLoader]:
    full_ds = OpenIDataset(data_dir, img_size=img_size, text_model=text_model, augment=True)
    n = len(full_ds)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
