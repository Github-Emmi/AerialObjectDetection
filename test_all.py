#!/usr/bin/env python3
"""Pre-deployment verification suite for Aerial Object Detection project."""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
RESULTS = []


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        tag = "PASS"
    else:
        FAIL += 1
        tag = "FAIL"
    msg = f"  [{tag}] {name}" + (f" — {detail}" if detail else "")
    print(msg)
    RESULTS.append((tag, name, detail))


# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Essential Files & Model Weights")
print("=" * 60)

essential_files = [
    "app/app.py", "app/components/classifier_ui.py",
    "app/components/detector_ui.py", "app/components/__init__.py",
    "src/__init__.py", "src/config.py", "src/models/__init__.py",
    "src/models/custom_cnn.py", "src/models/transfer_learning.py",
    "src/models/yolo_detector.py", "src/preprocessing.py",
    "src/evaluate.py", "src/utils.py", "data.yaml",
    "requirements.txt", "Dockerfile", "docker-compose.yml",
    ".dockerignore", "README.md",
]
weight_files = [
    ("models/classification/custom_cnn/best_model.pth", 1.0),
    ("models/classification/resnet50/best_model.pth", 80.0),
    ("models/classification/mobilenet_v2/best_model.pth", 5.0),
    ("models/classification/efficientnet_b0/best_model.pth", 10.0),
    ("models/detection/best.pt", 40.0),
]

for f in essential_files:
    p = ROOT / f
    report(f"File: {f}", p.exists(), f"{p.stat().st_size} bytes" if p.exists() else "NOT FOUND")

for f, min_mb in weight_files:
    p = ROOT / f
    if p.exists():
        sz_mb = p.stat().st_size / 1024 / 1024
        ok = sz_mb >= min_mb
        report(f"Weight: {f}", ok, f"{sz_mb:.1f} MB (min {min_mb} MB)")
    else:
        report(f"Weight: {f}", False, "NOT FOUND")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Module Imports")
print("=" * 60)

modules = [
    "src.config",
    "src.models.custom_cnn",
    "src.models.transfer_learning",
    "src.models.yolo_detector",
    "src.preprocessing",
    "src.evaluate",
    "src.utils",
    "src.data_validation",
]

for mod in modules:
    try:
        __import__(mod)
        report(f"Import: {mod}", True)
    except Exception as e:
        report(f"Import: {mod}", False, str(e))

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Classification Model Loading & Inference")
print("=" * 60)

import torch
from src.models.custom_cnn import AerialCNN
from src.models.transfer_learning import create_transfer_model
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Create a dummy image tensor
dummy = torch.randn(1, 3, 224, 224)

classifiers = {
    "custom_cnn": lambda: AerialCNN(num_classes=2),
    "resnet50": lambda: create_transfer_model("resnet50", 2, 0.0),
    "mobilenet_v2": lambda: create_transfer_model("mobilenet_v2", 2, 0.0),
    "efficientnet_b0": lambda: create_transfer_model("efficientnet_b0", 2, 0.0),
}

for name, factory in classifiers.items():
    try:
        model = factory()
        weights_path = ROOT / "models" / "classification" / name / "best_model.pth"
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
            probs = torch.softmax(out, dim=1).squeeze()
        bird, drone = probs[0].item(), probs[1].item()
        report(f"Classifier: {name}", True, f"Bird={bird:.3f} Drone={drone:.3f}")
    except Exception as e:
        report(f"Classifier: {name}", False, traceback.format_exc().split('\n')[-2])

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: YOLOv8m Detection Model Loading & Inference")
print("=" * 60)

try:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np

    yolo = YOLO(str(ROOT / "models" / "detection" / "best.pt"))
    report("YOLOv8m: load weights", True, f"model type: {yolo.task}")

    # Inference on synthetic image
    synth = Image.fromarray(np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8))
    res = yolo.predict(synth, conf=0.1, verbose=False)
    report("YOLOv8m: synthetic inference", True, f"{len(res[0].boxes)} boxes")

    # Inference on real test image (if available)
    test_imgs = ROOT / "test" / "images"
    if test_imgs.exists():
        real_img = next(test_imgs.iterdir(), None)
        if real_img:
            res2 = yolo.predict(str(real_img), conf=0.25, verbose=False)
            report("YOLOv8m: real image inference", True,
                   f"{len(res2[0].boxes)} detections on {real_img.name}")
        else:
            report("YOLOv8m: real image inference", False, "No images in test/images/")
    else:
        report("YOLOv8m: real image inference", False, "test/images/ not found")

except Exception as e:
    report("YOLOv8m", False, traceback.format_exc().split('\n')[-2])

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 5: Streamlit App Syntax & Import Check")
print("=" * 60)

try:
    import ast
    for app_file in ["app/app.py", "app/components/classifier_ui.py", "app/components/detector_ui.py"]:
        with open(ROOT / app_file) as f:
            ast.parse(f.read(), filename=app_file)
        report(f"Syntax: {app_file}", True)
except SyntaxError as e:
    report(f"Syntax: {app_file}", False, str(e))

try:
    import streamlit
    report("Import: streamlit", True, f"v{streamlit.__version__}")
except ImportError as e:
    report("Import: streamlit", False, str(e))

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 6: data.yaml Integrity")
print("=" * 60)

try:
    import yaml
    with open(ROOT / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    report("data.yaml: parse", True)
    report("data.yaml: nc==2", cfg.get("nc") == 2, f"nc={cfg.get('nc')}")
    report("data.yaml: names", len(cfg.get("names", [])) == 2, f"names={cfg.get('names')}")
    report("data.yaml: train path", "train" in cfg, f"train={cfg.get('train')}")
    report("data.yaml: val path", "val" in cfg, f"val={cfg.get('val')}")
    report("data.yaml: test path", "test" in cfg, f"test={cfg.get('test')}")
except Exception as e:
    report("data.yaml", False, str(e))

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 7: Docker Configuration Checks")
print("=" * 60)

# Verify Dockerfile references match actual files
try:
    dockerfile = (ROOT / "Dockerfile").read_text()
    checks = [
        ("COPY requirements.txt", "requirements.txt" in dockerfile),
        ("COPY src/", "COPY src/" in dockerfile),
        ("COPY app/", "COPY app/" in dockerfile),
        ("COPY models/", "COPY models/" in dockerfile),
        ("COPY data.yaml", "COPY data.yaml" in dockerfile),
        ("HEALTHCHECK", "HEALTHCHECK" in dockerfile),
        ("EXPOSE 8501", "8501" in dockerfile),
        ("curl in apt-get", "curl" in dockerfile),
    ]
    for name, ok in checks:
        report(f"Dockerfile: {name}", ok)
except Exception as e:
    report("Dockerfile", False, str(e))

try:
    with open(ROOT / "docker-compose.yml") as f:
        dc = yaml.safe_load(f)
    report("docker-compose.yml: parse", True)
    ports = dc.get("services", {}).get("app", {}).get("ports", [])
    report("docker-compose.yml: port 8501", "8501:8501" in ports, f"ports={ports}")
except Exception as e:
    report("docker-compose.yml", False, str(e))

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 8: .dockerignore Whitelist Verification")
print("=" * 60)

try:
    di = (ROOT / ".dockerignore").read_text()
    needed = ["!app/**", "!src/**", "!requirements.txt", "!data.yaml",
              "!models/classification", "!models/detection"]
    for item in needed:
        report(f".dockerignore: {item}", item in di)
except Exception as e:
    report(".dockerignore", False, str(e))

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"SUMMARY: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
print("=" * 60)
if FAIL > 0:
    print("\nFailed tests:")
    for tag, name, detail in RESULTS:
        if tag == "FAIL":
            print(f"  ✗ {name}: {detail}")
    sys.exit(1)
else:
    print("\n  All tests passed! Ready for deployment.")
    sys.exit(0)
