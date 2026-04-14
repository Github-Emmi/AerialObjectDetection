"""Central configuration for the Aerial Object Classification & Detection project."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
DATA_YAML = PROJECT_ROOT / "data.yaml"
DETECTION_ROOT = PROJECT_ROOT
CLASSIFICATION_ROOT = PROJECT_ROOT / "classification_dataset"

# Class mapping (from data.yaml)
CLASS_NAMES = {0: "Bird", 1: "drone"}
NUM_CLASSES = 2

# Split names
SPLITS = ("train", "valid", "test")


@dataclass
class ClassificationConfig:
    data_root: Path = field(default_factory=lambda: CLASSIFICATION_ROOT)
    model_save_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "classification")
    input_size: int = 224
    batch_size: int = 32
    num_classes: int = NUM_CLASSES
    learning_rate: float = 1e-3
    transfer_lr: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    scheduler: str = "cosine"
    weight_decay: float = 1e-4


@dataclass
class DetectionConfig:
    data_yaml: Path = field(default_factory=lambda: DATA_YAML)
    model_save_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "detection")
    model_variant: str = "yolov8m.pt"
    imgsz: int = 416
    batch_size: int = 16
    epochs: int = 100
    patience: int = 20
    optimizer: str = "AdamW"
    lr0: float = 1e-3
    weight_decay: float = 5e-4
