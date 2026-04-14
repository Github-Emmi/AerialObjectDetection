"""YOLOv8m detection training script.

Usage:
    python -m src.train_detector
    python -m src.train_detector --epochs 50 --batch-size 8
"""

import argparse

from src.config import DetectionConfig
from src.models.yolo_detector import train_yolov8


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8m aerial detector")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()

    cfg = DetectionConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz

    train_yolov8(cfg)


if __name__ == "__main__":
    main()
