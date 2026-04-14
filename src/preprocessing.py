"""Data preprocessing and augmentation pipelines.

Classification: torchvision transforms + ImageFolder dataloaders (224×224).
Detection: YOLOv8 handles its own augmentation internally — this module
provides helper config dicts and the updated data.yaml path only.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.config import (
    CLASSIFICATION_ROOT,
    DETECTION_ROOT,
    SPLITS,
    ClassificationConfig,
    DetectionConfig,
)

# ──────────────────────────────────────────────────────────────
# ImageNet normalisation (used for both custom CNN & pretrained)
# ──────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_classification_transforms(cfg: ClassificationConfig | None = None):
    """Return a dict of train / valid / test transforms."""
    size = cfg.input_size if cfg else 224
    return {
        "train": transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        "valid": transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        "test": transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    }


def _is_valid_image(path: str) -> bool:
    """Filter out macOS ._ resource-fork files from ImageFolder."""
    return not Path(path).name.startswith("._")


def get_classification_loaders(
    cfg: ClassificationConfig | None = None,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Build DataLoaders for classification train / valid / test splits.

    Returns
    -------
    dict  mapping split name → DataLoader
    """
    if cfg is None:
        cfg = ClassificationConfig()

    tx = get_classification_transforms(cfg)
    loaders: dict[str, DataLoader] = {}

    for split in SPLITS:
        split_dir = cfg.data_root / split
        if not split_dir.exists():
            continue

        dataset = ImageFolder(
            root=str(split_dir),
            transform=tx[split],
            is_valid_file=_is_valid_image,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return loaders


# ──────────────────────────────────────────────────────────────
# Detection helpers (YOLOv8 handles transforms internally)
# ──────────────────────────────────────────────────────────────


def get_detection_train_args(cfg: DetectionConfig | None = None) -> dict:
    """Return the keyword args dict to pass to ``model.train(**args)``.

    YOLOv8 manages its own augmentation pipeline — this function
    centralises the hyper-parameters so they live next to the
    classification transforms.
    """
    if cfg is None:
        cfg = DetectionConfig()

    return {
        "data": str(cfg.data_yaml),
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch_size,
        "project": str(cfg.model_save_dir),
        "name": "yolov8m_aerial",
        "patience": cfg.patience,
        "save": True,
        "save_period": 10,
        "device": "0" if torch.cuda.is_available() else "cpu",
        "workers": 4,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "lrf": 0.01,
        "weight_decay": cfg.weight_decay,
        "warmup_epochs": 3,
        "cos_lr": True,
        "plots": True,
        "verbose": True,
        # Augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
    }
