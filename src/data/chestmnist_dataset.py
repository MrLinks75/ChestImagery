from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import ChestMNIST


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_transforms(img_size: int, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_chestmnist_loaders(img_size: int = 128, batch_size: int = 64,
                            num_workers: int = 4) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = ChestMNIST(split="train", transform=_get_transforms(img_size, augment=True),
                           download=True, size=img_size, as_rgb=True)
    val_ds = ChestMNIST(split="val", transform=_get_transforms(img_size, augment=False),
                         download=True, size=img_size, as_rgb=True)
    test_ds = ChestMNIST(split="test", transform=_get_transforms(img_size, augment=False),
                          download=True, size=img_size, as_rgb=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def compute_pos_weights(train_loader: DataLoader, num_classes: int = 14) -> torch.Tensor:
    """BCE positive weights to counter class imbalance."""
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels.float())
    all_labels = torch.cat(all_labels, dim=0)
    pos = all_labels.sum(0)
    neg = len(all_labels) - pos
    pos_weight = neg / pos.clamp(min=1)
    return pos_weight


def get_normal_indices(dataset) -> list[int]:
    """Returns indices of samples with no positive pathology label."""
    normal = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            if label.sum() == 0:
                normal.append(i)
        elif isinstance(label, np.ndarray):
            if label.sum() == 0:
                normal.append(i)
    return normal
