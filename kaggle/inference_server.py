"""FastAPI inference server for Aerial Object Classification & Detection.

Runs on Kaggle GPU (Tesla T4/P100) and is exposed via Ngrok tunnel.
Serves both classification and detection endpoints.
"""

import io
import os
import base64
import time

import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

# ── Paths (Kaggle dataset mount) ───────────────────────────────
DATASET_ROOT = "/kaggle/input/aerial-bird-drone-detection"
CLASSIFICATION_DIR = os.path.join(DATASET_ROOT, "models", "classification")
DETECTION_WEIGHTS = os.path.join(DATASET_ROOT, "models", "detection", "best.pt")

CLASS_NAMES = ["Bird", "Drone"]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── FastAPI app ─────────────────────────────────────────────────
app = FastAPI(title="Aerial Detection API", version="1.0.0")

# ── Global model cache ──────────────────────────────────────────
_models = {}
_device = None


def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {_device}")
    return _device


def _load_custom_cnn():
    """Load the 4-block Custom CNN."""
    import torch.nn as nn

    class AerialCNN(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
                nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = AerialCNN(num_classes=2)
    weights = os.path.join(CLASSIFICATION_DIR, "custom_cnn", "best_model.pth")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def _load_transfer_model(backbone: str):
    """Load a transfer-learning model (resnet50, mobilenet_v2, efficientnet_b0)."""
    import torch.nn as nn

    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, 2))
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, 2))
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, 2))
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    weights = os.path.join(CLASSIFICATION_DIR, backbone, "best_model.pth")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def get_classifier(backbone: str):
    """Return cached classifier, loading on first call."""
    if backbone not in _models:
        device = get_device()
        if backbone == "custom_cnn":
            model = _load_custom_cnn()
        else:
            model = _load_transfer_model(backbone)
        model.to(device).eval()
        _models[backbone] = model
        print(f"[INFO] Loaded classifier: {backbone}")
    return _models[backbone]


def get_detector():
    """Return cached YOLOv8m detector, loading on first call."""
    if "yolov8m" not in _models:
        from ultralytics import YOLO
        model = YOLO(DETECTION_WEIGHTS)
        _models["yolov8m"] = model
        print("[INFO] Loaded YOLOv8m detector")
    return _models["yolov8m"]


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "device": device,
        "models_loaded": list(_models.keys()),
        "timestamp": time.time(),
    }


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    backbone: str = Query("mobilenet_v2", description="Model backbone"),
):
    """Classify an aerial image as Bird or Drone."""
    t0 = time.time()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    device = get_device()
    model = get_classifier(backbone)
    tensor = PREPROCESS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu()

    pred_idx = probs.argmax().item()
    result = {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
        "backbone": backbone,
        "device": str(device),
        "inference_ms": round((time.time() - t0) * 1000, 1),
    }
    return JSONResponse(content=result)


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.1, le=1.0, description="Confidence threshold"),
):
    """Detect birds and drones with bounding boxes using YOLOv8m."""
    t0 = time.time()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    model = get_detector()
    results = model.predict(image, conf=confidence, verbose=False)
    result = results[0]

    # Build detections list
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class": result.names[cls_id],
            "confidence": round(conf, 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    # Encode annotated image as base64 PNG
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]
    annotated_pil = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = {
        "detections": detections,
        "count": len(detections),
        "annotated_image_b64": img_b64,
        "confidence_threshold": confidence,
        "device": str(get_device()),
        "inference_ms": round((time.time() - t0) * 1000, 1),
    }
    return JSONResponse(content=response)
