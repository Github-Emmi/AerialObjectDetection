"""Classification training loop with early stopping and TensorBoard logging.

Usage:
    python -m src.train_classifier --model custom_cnn
    python -m src.train_classifier --model resnet50
    python -m src.train_classifier --model mobilenet_v2
    python -m src.train_classifier --model efficientnet_b0
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from src.config import ClassificationConfig, PROJECT_ROOT
from src.models.custom_cnn import AerialCNN
from src.models.transfer_learning import create_transfer_model
from src.preprocessing import get_classification_loaders

SEED = 42


def _set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(name: str, num_classes: int) -> nn.Module:
    if name == "custom_cnn":
        return AerialCNN(num_classes=num_classes)
    return create_transfer_model(backbone=name, num_classes=num_classes)


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


def train_classifier(model_name: str = "custom_cnn", cfg: ClassificationConfig | None = None):
    _set_seed()
    if cfg is None:
        cfg = ClassificationConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    loaders = get_classification_loaders(cfg, num_workers=4)
    print(f"Train: {len(loaders['train'].dataset)} | "
          f"Valid: {len(loaders['valid'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # Model
    lr = cfg.transfer_lr if model_name != "custom_cnn" else cfg.learning_rate
    model = _build_model(model_name, cfg.num_classes).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Params: {total:,} (trainable: {trainable:,})")

    # Optimiser, loss, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Logging
    log_dir = PROJECT_ROOT / "runs" / model_name
    writer = SummaryWriter(log_dir=str(log_dir))

    # Checkpoint directory
    save_dir = cfg.model_save_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["valid"], criterion, device)
        scheduler.step()

        writer.add_scalars("Loss", {"train": train_loss, "valid": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "valid": val_acc}, epoch)

        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s | Best val acc: {best_val_acc:.4f}")
    print(f"Model saved to {save_dir / 'best_model.pth'}")

    writer.close()
    return model


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train aerial classifier")
    parser.add_argument(
        "--model", type=str, default="custom_cnn",
        choices=["custom_cnn", "resnet50", "mobilenet_v2", "efficientnet_b0"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = ClassificationConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
        cfg.transfer_lr = args.lr

    train_classifier(args.model, cfg)


if __name__ == "__main__":
    main()
