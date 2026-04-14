#!/usr/bin/env python3
"""
Aerial YOLOv8m Detection Training — Kaggle GPU Script
======================================================
Self-contained training script for the Aerial Bird vs Drone object detector.
Runs on Kaggle with GPU enabled. Uses Ultralytics YOLOv8m.

Dataset mount path on Kaggle: /kaggle/input/aerial-bird-drone-detection/
Output path: /kaggle/working/

Usage (Kaggle):
    Pushed via `kaggle kernels push` — runs automatically.

Usage (local test with 2 epochs):
    python kaggle/train_detection_kaggle.py --epochs 2 --local
"""

import argparse
import os
import shutil
import subprocess
import sys
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

        # 4. Reinstall ultralytics (depends on torch, may have been broken)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "ultralytics",
        ])

        # 5. Downgrade numpy LAST — ultralytics pulls numpy>=2 but
        #    torch 2.2.2 was compiled against numpy 1.x
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "numpy<2",
        ])
        print("Compatible PyTorch + NumPy + Ultralytics installed successfully!")

    except Exception as e:
        print(f"GPU compatibility check skipped: {e}")


ensure_gpu_compatible_pytorch()

import torch
import yaml


def create_kaggle_data_yaml(data_root: Path, output_dir: Path) -> Path:
    """Create a data.yaml pointing to Kaggle paths."""
    yaml_content = {
        "path": str(data_root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["Bird", "drone"],
    }
    yaml_path = output_dir / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Aerial YOLOv8m Detection Training (Kaggle GPU)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="YOLOv8 pretrained model")
    parser.add_argument("--local", action="store_true",
                        help="Use local paths instead of Kaggle mount paths")
    args = parser.parse_args()

    # Determine paths
    if args.local:
        data_root = Path(__file__).resolve().parent.parent
        output_dir = Path(__file__).resolve().parent.parent / "models" / "detection"
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

        # Try multiple possible Kaggle mount paths
        candidates = [
            kaggle_input / "aerial-bird-drone-detection",
            kaggle_input / "datasets" / "aghasonemmanuel" / "aerial-bird-drone-detection",
        ]
        data_root = None
        for c in candidates:
            if c.exists() and (c / "train" / "images").exists():
                data_root = c
                break
        if data_root is None:
            # Last resort: search for train/images anywhere under /kaggle/input
            for p in kaggle_input.rglob("train/images"):
                data_root = p.parent.parent
                break
        if data_root is None:
            data_root = candidates[0]  # Will fail with clear error below

        output_dir = Path("/kaggle/working/detection")

    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Verify dataset structure
    train_images = data_root / "train" / "images"
    valid_images = data_root / "valid" / "images"
    if not train_images.exists() or not valid_images.exists():
        print(f"ERROR: Dataset structure not found at {data_root}")
        print(f"  Expected: {train_images}")
        print(f"  Expected: {valid_images}")
        print("On Kaggle, ensure the dataset 'aerial-bird-drone-detection' is attached.")
        sys.exit(1)

    train_count = len(list(train_images.glob("*")))
    valid_count = len(list(valid_images.glob("*")))
    print(f"Train images: {train_count} | Valid images: {valid_count}")

    # Create data.yaml for this run
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = create_kaggle_data_yaml(data_root, output_dir)
    print(f"Data YAML: {yaml_path}")

    # Import and run ultralytics
    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"Model: {args.model}")

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(output_dir),
        name="yolov8m_aerial",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best weights to output root for easy download
    best_weights = output_dir / "yolov8m_aerial" / "weights" / "best.pt"
    if best_weights.exists():
        dest = output_dir / "best.pt"
        shutil.copy2(best_weights, dest)
        print(f"\nBest weights copied to: {dest}")

    # Run validation on test set
    print("\nRunning evaluation on test set...")
    best_model = YOLO(str(best_weights) if best_weights.exists() else args.model)
    test_images = data_root / "test" / "images"
    if test_images.exists():
        metrics = best_model.val(
            data=str(yaml_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=0 if torch.cuda.is_available() else "cpu",
            project=str(output_dir),
            name="yolov8m_test_eval",
            exist_ok=True,
        )
        print(f"\nTest mAP50: {metrics.box.map50:.4f}")
        print(f"Test mAP50-95: {metrics.box.map:.4f}")

        # Save test results
        with open(output_dir / "test_results.txt", "w") as f:
            f.write(f"model=yolov8m\n")
            f.write(f"test_mAP50={metrics.box.map50:.4f}\n")
            f.write(f"test_mAP50_95={metrics.box.map:.4f}\n")
            f.write(f"epochs={args.epochs}\n")
            f.write(f"imgsz={args.imgsz}\n")
            f.write(f"batch={args.batch}\n")
    else:
        print(f"WARNING: Test images not found at {test_images}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
