#!/usr/bin/env python3
"""
Aerial Classification Training — Kaggle GPU Script
====================================================
Self-contained training script for the Aerial Bird vs Drone classifier.
Runs on Kaggle with GPU enabled. Trains all 4 models sequentially:
  1. Custom CNN (~422K params)
  2. ResNet50 (~23.5M params, 80% frozen)
  3. MobileNetV2 (~2.2M params, 80% frozen)
  4. EfficientNet-B0 (~4.0M params, 80% frozen)

Dataset mount path on Kaggle: /kaggle/input/aerial-bird-drone-detection/
Output path: /kaggle/working/

Usage (Kaggle):
    Pushed via `kaggle kernels push` — runs automatically.

Usage (local test with 2 epochs):
    python kaggle/train_classification_kaggle.py --epochs 2 --local
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def ensure_gpu_compatible_pytorch():
    """Detect P100 GPU via nvidia-smi (before importing torch) and install
    a compatible PyTorch build.  Old packages are fully uninstalled first
    so no stale torchvision files remain."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or "P100" not in result.stdout:
            print(f"GPU: {result.stdout.strip()} — using pre-installed PyTorch")
            return

        print(f"Detected P100 GPU: {result.stdout.strip()}")
        print("Uninstalling incompatible PyTorch stack...")

        # 1. Uninstall old packages
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y",
             "torch", "torchvision", "torchaudio"],
            capture_output=True,
        )

        # 2. Remove any leftover dist-packages dirs (pip uninstall can miss files)
        site_packages = Path("/usr/local/lib/python3.12/dist-packages")
        for pkg in ["torch", "torchvision", "torchaudio"]:
            pkg_dir = site_packages / pkg
            if pkg_dir.exists():
                shutil.rmtree(pkg_dir, ignore_errors=True)
                print(f"  Cleaned {pkg_dir}")

        # 3. Install compatible versions
        print("Installing PyTorch 2.2.2+cu118 for P100 compatibility...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.2.2", "torchvision==0.17.2",
            "--index-url", "https://download.pytorch.org/whl/cu118",
        ])

        # 4. Downgrade numpy — torch 2.2.2 was compiled against numpy 1.x
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "numpy<2",
        ])
        print("Compatible PyTorch + NumPy installed successfully!")

    except Exception as e:
        print(f"GPU compatibility check skipped: {e}")


ensure_gpu_compatible_pytorch()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# ── Constants ────────────────────────────────────────────────
SEED = 42
NUM_CLASSES = 2
CLASS_NAMES = ["Bird", "drone"]
INPUT_SIZE = 224
BATCH_SIZE = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Models to train sequentially
MODELS_TO_TRAIN = ["custom_cnn", "resnet50", "mobilenet_v2", "efficientnet_b0"]


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


# ── Model Definitions ────────────────────────────────────────

class AerialCNN(nn.Module):
    """Custom 4-block CNN for Bird vs Drone classification (224x224 input)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_transfer_model(backbone: str, num_classes: int = 2, freeze_ratio: float = 0.8):
    """Create a pretrained model with frozen backbone layers."""
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, num_classes)) # type: ignore
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, num_classes))
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, num_classes))
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    params = list(model.parameters())
    freeze_count = int(len(params) * freeze_ratio)
    for param in params[:freeze_count]:
        param.requires_grad = False

    return model


def build_model(name: str) -> nn.Module:
    if name == "custom_cnn":
        return AerialCNN(num_classes=NUM_CLASSES)
    return create_transfer_model(backbone=name, num_classes=NUM_CLASSES)


# ── Data Loading ─────────────────────────────────────────────

def is_valid_image(path: str) -> bool:
    """Filter out macOS ._ resource-fork files."""
    return not Path(path).name.startswith("._")


def get_transforms():
    return {
        "train": transforms.Compose([
            transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        "valid": transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        "test": transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    }


def get_loaders(data_root: Path, batch_size: int = BATCH_SIZE):
    tx = get_transforms()
    loaders = {}
    for split in ["train", "valid", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"WARNING: {split_dir} not found, skipping")
            continue
        dataset = ImageFolder(root=str(split_dir), transform=tx.get(split, tx["valid"]),
                              is_valid_file=is_valid_image)
        loaders[split] = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    num_workers=2, pin_memory=torch.cuda.is_available(),
                                    drop_last=(split == "train"))
    return loaders


# ── Training Loop ────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train_model(model_name: str, data_root: Path, output_dir: Path,
                epochs: int = 50, patience: int = 10):
    """Train a single model and save best weights + log."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: {model_name} | Device: {device} | Epochs: {epochs}")
    print(f"{'='*60}")

    loaders = get_loaders(data_root)
    if "train" not in loaders or "valid" not in loaders:
        print(f"ERROR: Missing train or valid split in {data_root}")
        return None

    print(f"Train: {len(loaders['train'].dataset)} images | "
          f"Valid: {len(loaders['valid'].dataset)} images")

    model = build_model(model_name).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,} total | {trainable:,} trainable")

    lr = 1e-4 if model_name != "custom_cnn" else 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = output_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    log_rows = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"],
                                                criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["valid"], criterion, device)
        scheduler.step()

        log_rows.append({
            "epoch": epoch, "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.4f}", "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
        })

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s | Best val acc: {best_val_acc:.4f}")

    # Save training log
    log_path = save_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                                "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(log_rows)

    # Test evaluation
    if "test" in loaders:
        test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
        print(f"Test Accuracy: {test_acc:.4f}")
        with open(save_dir / "test_results.txt", "w") as f:
            f.write(f"model={model_name}\n")
            f.write(f"test_accuracy={test_acc:.4f}\n")
            f.write(f"test_loss={test_loss:.4f}\n")
            f.write(f"best_val_accuracy={best_val_acc:.4f}\n")
            f.write(f"training_time_seconds={elapsed:.1f}\n")
            f.write(f"epochs_run={len(log_rows)}\n")
            f.write(f"device={device}\n")

    return best_val_acc


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aerial Classification Training (Kaggle GPU)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=MODELS_TO_TRAIN,
                        choices=MODELS_TO_TRAIN,
                        help="Which models to train (default: all 4)")
    parser.add_argument("--local", action="store_true",
                        help="Use local paths instead of Kaggle mount paths")
    args = parser.parse_args()

    # Determine paths
    if args.local:
        # Local development: repo root
        data_root = Path(__file__).resolve().parent.parent / "classification_dataset"
        output_dir = Path(__file__).resolve().parent.parent / "models" / "classification"
    else:
        # Kaggle environment — discover the actual mount path
        kaggle_input = Path("/kaggle/input")
        print(f"\n--- Kaggle /kaggle/input contents ---")
        if kaggle_input.exists():
            for item in sorted(kaggle_input.iterdir()):
                print(f"  {item.name}/" if item.is_dir() else f"  {item.name}")
                if item.is_dir():
                    for sub in sorted(item.iterdir())[:10]:
                        print(f"    {sub.name}/" if sub.is_dir() else f"    {sub.name}")
        print("--- end ---\n")

        # Try multiple possible paths for classification_dataset
        candidates = [
            kaggle_input / "aerial-bird-drone-detection" / "classification_dataset",
            kaggle_input / "aerial-bird-drone-detection" / "classification-dataset",
            kaggle_input / "aerial-bird-drone-detection",
        ]
        data_root = None
        for c in candidates:
            if c.exists() and (c / "train").exists():
                data_root = c
                break
        if data_root is None:
            # Last resort: search for a 'train' dir with bird/drone subdirs
            for p in kaggle_input.rglob("train/bird"):
                data_root = p.parent.parent
                break
        if data_root is None:
            data_root = candidates[0]  # Will fail with clear error below

        output_dir = Path("/kaggle/working/classification")

    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Verify dataset exists
    if not data_root.exists():
        print(f"ERROR: Data root not found: {data_root}")
        print("On Kaggle, ensure the dataset 'aerial-bird-drone-detection' is attached.")
        sys.exit(1)

    # Train each model
    summary = {}
    for model_name in args.models:
        acc = train_model(model_name, data_root, output_dir,
                          epochs=args.epochs, patience=args.patience)
        if acc is not None:
            summary[model_name] = acc

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, acc in summary.items():
        print(f"  {name:20s}: {acc:.4f} best val accuracy")

    # Save summary
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.txt", "w") as f:
        for name, acc in summary.items():
            f.write(f"{name},{acc:.4f}\n")

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
