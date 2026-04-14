"""YOLOv8m wrapper for aerial object detection."""

from pathlib import Path

from ultralytics import YOLO

from src.config import DetectionConfig
from src.preprocessing import get_detection_train_args


def train_yolov8(cfg: DetectionConfig | None = None) -> YOLO:
    """Train YOLOv8m on the aerial detection dataset.

    Returns the trained YOLO model instance.
    """
    if cfg is None:
        cfg = DetectionConfig()

    model = YOLO(cfg.model_variant)
    args = get_detection_train_args(cfg)
    model.train(**args)
    return model


def load_yolov8(weights_path: str | Path) -> YOLO:
    """Load a trained YOLOv8 model from weights."""
    return YOLO(str(weights_path))
