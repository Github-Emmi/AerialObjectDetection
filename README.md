# Aerial Object Classification & Detection

A deep learning pipeline that **classifies** aerial images as Bird or Drone and **detects + localizes** them with bounding boxes. Built with PyTorch and YOLOv8, deployed via Streamlit.

---

## Key Results

| Model | Task | Accuracy / mAP@50 | F1 | Params |
|-------|------|-------------------:|---:|-------:|
| Custom CNN | Classification | 84.2% | 0.840 | 423K |
| ResNet50 | Classification | 97.7% | 0.977 | 23.5M |
| **MobileNetV2** | Classification | **98.6%** | **0.986** | **2.2M** |
| EfficientNet-B0 | Classification | 98.6% | 0.986 | 4.0M |
| YOLOv8m | Detection | 81.3% mAP@50 | — | 25.8M |

**Best classifier**: MobileNetV2 — highest F1 (0.986) with smallest footprint (2.2M params, 63ms/image).

**Detection per-class (test)**: Bird AP@50 = 0.718 · Drone AP@50 = 0.908

---

## Project Structure

```
├── app/                        # Streamlit application
│   ├── app.py                  #   Main entry point
│   └── components/
│       ├── classifier_ui.py    #   Classification mode
│       └── detector_ui.py      #   Detection mode
├── src/
│   ├── config.py               # Central configuration
│   ├── evaluate.py             # Unified evaluation pipeline
│   ├── utils.py                # Plotting utilities
│   └── models/
│       ├── custom_cnn.py       # 4-block CNN architecture
│       └── transfer_learning.py # ResNet50, MobileNetV2, EfficientNet-B0
├── models/
│   ├── classification/         # Saved .pth weights (4 models)
│   └── detection/              # YOLOv8m best.pt + training logs
├── reports/                    # Evaluation reports, curves, confusion matrices
├── classification_dataset/     # Bird/Drone image folders (train/valid/test)
├── train/ valid/ test/         # YOLO-format detection splits (images + labels)
├── data.yaml                   # YOLO dataset config
├── Dockerfile                  # Container build
├── docker-compose.yml          # One-command deployment
└── requirements.txt            # Python dependencies
```

---

## Quick Start

### 1. Local Setup

```bash
# Clone & enter
git clone <repo-url> && cd object_detection_Dataset

# Create environment (Python 3.12)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Launch the app
streamlit run app/app.py
```

Open **http://localhost:8501** → upload an image → classify or detect.

### 2. Docker

```bash
docker compose up --build
```

Open **http://localhost:8501**.

---

## Dataset

| Split | Classification | Detection |
|-------|---------------:|----------:|
| Train | 2,662 | 2,728 |
| Valid | 442 | 448 |
| Test | 215 | 224 |

- **Classes**: Bird (0), Drone (1)
- **Source**: Roboflow export, YOLO-format annotations
- Classification dataset is the detection dataset reorganized into class folders (81 background images excluded)
- Bbox-level imbalance: Bird 3,157 : Drone 1,702 (1.9:1)

---

## Training

All models were trained on **Kaggle P100 GPU**.

### Classification (4 models)
- Input: 224×224, ImageNet normalization
- Optimizer: Adam (Custom CNN: lr=1e-3) / AdamW (transfer: lr=1e-4)
- Early stopping: patience=10, cosine LR scheduler
- Augmentation: rotation, flip, color jitter, random crop

### Detection (YOLOv8m)
- Input: 416×416
- 86 epochs (early stopping at patience=20)
- Optimizer: AdamW, lr0=1e-3
- Training time: 1.18 hours on P100

---

## Evaluation

Run the unified evaluation pipeline:

```bash
python -m src.evaluate                    # All models
python -m src.evaluate --detection-only   # Detection only
python -m src.evaluate --skip-detection   # Classification only
```

Outputs go to `reports/`: confusion matrices, ROC curves, training curves, `model_comparison.csv`.

---

## Streamlit App

Two modes accessible from the sidebar:

**Classification** — Upload an image, select any of the 4 classifiers from a dropdown, view the predicted class and confidence bar chart.

**Detection** — Upload an image, adjust the confidence threshold slider, view the annotated image with bounding boxes and a detection summary table.

---

## Tech Stack

- **Python 3.12** · **PyTorch 2.2** · **torchvision 0.17**
- **Ultralytics YOLOv8** (object detection)
- **Streamlit** (web UI)
- **scikit-learn** (metrics) · **matplotlib / seaborn** (visualization)
- **Docker** (containerization)

---

## License

This project uses the [Roboflow dataset](https://universe.roboflow.com/) — refer to `README.roboflow.txt` for dataset license terms.
