# 🛩️ Aerial Object Classification & Detection

A deep learning solution that **classifies** aerial images as **Bird** or **Drone** and **detects + localizes** them with bounding boxes in real-world scenes. Built with PyTorch, YOLOv8, and deployed via Streamlit & Docker.

> **Built by [Aghason Emmanuel Ibeabuchi](https://github.com/Github-Emmi) (Emmi Dev Codes) — Lagos, Nigeria**

---

## 📌 Problem Statement

Accurate identification between drones and birds in aerial imagery is critical for **security surveillance**, **wildlife protection**, and **airspace safety**. This project develops a deep learning pipeline that:

1. **Classifies** aerial images into Bird or Drone using a Custom CNN and Transfer Learning models
2. **Detects & localizes** birds and drones with bounding boxes using YOLOv8m
3. **Deploys** an interactive web application for real-time inference

### Real-World Use Cases

- **Wildlife Protection** — Detect birds near wind farms or airports to prevent collisions
- **Security & Defense** — Identify drones in restricted airspace for timely alerts
- **Airport Bird-Strike Prevention** — Monitor runway zones for bird activity
- **Environmental Research** — Track bird populations using aerial footage

---

## 🏆 Key Results

| Model | Task | Accuracy / mAP@50 | F1 Score | ROC-AUC | Params | Inference |
|-------|------|-------------------:|:--------:|:-------:|-------:|----------:|
| Custom CNN | Classification | 84.2% | 0.840 | 0.920 | 423K | 58 ms |
| ResNet50 | Classification | 97.7% | 0.977 | 0.999 | 23.5M | 175 ms |
| **MobileNetV2** | **Classification** | **98.6%** | **0.986** | **0.999** | **2.2M** | **63 ms** |
| EfficientNet-B0 | Classification | 98.6% | 0.986 | 1.000 | 4.0M | 68 ms |
| YOLOv8m | Detection | 81.3% mAP@50 | — | — | 25.8M | 5.3 ms |

**Best Classifier**: MobileNetV2 — highest F1 (0.986) with smallest footprint (2.2M params)

**Detection Per-Class**: Bird AP@50 = 0.718 · Drone AP@50 = 0.908

---

## 📂 Project Structure

```
├── app/                            # Streamlit web application
│   ├── app.py                      #   Main entry point (sidebar navigation)
│   └── components/
│       ├── classifier_ui.py        #   Classification mode (4-model dropdown)
│       └── detector_ui.py          #   Detection mode (YOLOv8m + bbox overlay)
├── src/                            # Core source code
│   ├── config.py                   #   Central configuration & paths
│   ├── preprocessing.py            #   Transforms, data loaders, augmentation
│   ├── evaluate.py                 #   Unified evaluation pipeline (CLI)
│   ├── train_classifier.py         #   Classification training loop
│   ├── train_detector.py           #   YOLOv8m training wrapper
│   ├── data_validation.py          #   Dataset integrity checks
│   ├── utils.py                    #   Plotting utilities
│   └── models/
│       ├── custom_cnn.py           #   4-block CNN (Conv→BN→ReLU→Pool→Dropout)
│       ├── transfer_learning.py    #   ResNet50, MobileNetV2, EfficientNet-B0
│       └── yolo_detector.py        #   YOLOv8 train/load wrappers
├── models/                         # Trained model weights
│   ├── classification/             #   4× best_model.pth
│   └── detection/                  #   YOLOv8m best.pt
├── kaggle/                         # Kaggle GPU training scripts
│   ├── train_classification_kaggle.py
│   └── train_detection_kaggle.py
├── reports/                        # Evaluation outputs
│   ├── model_comparison.csv        #   Unified metrics table
│   ├── confusion_matrices/         #   Per-model confusion matrices
│   ├── roc_curves/                 #   ROC curve plots
│   └── training_curves/            #   Loss/accuracy over epochs
├── scripts/                        # Utility scripts
│   ├── validate_dataset.py         #   Run all dataset validation checks
│   ├── kaggle_push.sh              #   Push & poll Kaggle kernels
│   └── kaggle_pull_outputs.sh      #   Download trained weights
├── data.yaml                       # YOLO dataset configuration
├── Dockerfile                      # Docker container (CPU-only PyTorch)
├── docker-compose.yml              # One-command deployment
├── requirements.txt                # Pinned Python dependencies
├── test_all.py                     # Pre-deployment verification suite (65 tests)
└── README.md
```

---

## 📊 Datasets

### Classification Dataset

| Split | Bird | Drone | Total |
|-------|-----:|------:|------:|
| Train | 1,414 | 1,248 | 2,662 |
| Valid | 217 | 225 | 442 |
| Test | 121 | 94 | 215 |
| **Total** | **1,752** | **1,567** | **3,319** |

- **Task**: Binary Image Classification (Bird vs Drone)
- **Format**: RGB `.jpg` images, 224×224 resized
- **Source**: `classification_dataset/` directory

### Object Detection Dataset (YOLOv8 Format)

| Split | Images | Bird Boxes | Drone Boxes |
|-------|-------:|-----------:|------------:|
| Train | 2,728 | 2,533 | 1,364 |
| Valid | 448 | 411 | 216 |
| Test | 224 | 213 | 122 |
| **Total** | **3,400** | **3,157** | **1,702** |

- **Task**: Object Detection with bounding boxes
- **Annotation Format**: `<class_id> <x_center> <y_center> <width> <height>`
- **Class Imbalance**: Bird:Drone = 1.9:1
- **Source**: [Roboflow](https://universe.roboflow.com/) export

---

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# Clone the repository
git clone https://github.com/Github-Emmi/AerialObjectDetection.git
cd AerialObjectDetection

# Create virtual environment (Python 3.12)
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app/app.py
```

Open **http://localhost:8501** → upload an image → classify or detect.

### Option 2: Docker

```bash
docker compose up --build
```

Open **http://localhost:8501**. The Docker image uses CPU-only PyTorch (~2.5 GB) for efficient deployment.

### Option 3: Render (Cloud)

The app is deployed on Render using Docker. Visit the live demo:

> **[https://aerialobjectdetection.onrender.com](https://aerialobjectdetection.onrender.com)**

---

## 🔬 Project Workflow

### 1. Data Validation & Understanding
- Inspected dataset folder structure and image counts per class
- Identified class imbalance (Bird:Drone = 1.9:1 in detection)
- Cross-split duplicate check via MD5 hashing
- Label format verification for YOLO annotations
- Detected 81 empty label files (background images)

### 2. Data Preprocessing
- Normalized pixel values with ImageNet mean/std
- Resized images to 224×224 for classification, 416×416 for detection
- Built PyTorch `DataLoader` pipelines with macOS `._` file filtering

### 3. Data Augmentation
- **Classification**: Random crop, horizontal flip, rotation (±15°), color jitter (brightness, contrast, saturation)
- **Detection**: YOLOv8 built-in augmentation (mosaic, mixup, HSV augmentation)

### 4. Model Building

| Model | Architecture | Key Details |
|-------|-------------|-------------|
| **Custom CNN** | 4-block CNN | Conv→BN→ReLU→MaxPool→Dropout2d ×4, AdaptiveAvgPool, FC(256→128→2) |
| **ResNet50** | Transfer Learning | ImageNet pretrained, replaced FC head with Dropout(0.3)→Linear(2) |
| **MobileNetV2** | Transfer Learning | ImageNet pretrained, lightweight (2.2M params) |
| **EfficientNet-B0** | Transfer Learning | ImageNet pretrained, compound scaling |
| **YOLOv8m** | Object Detection | Medium variant, anchor-free, 25.8M params |

### 5. Model Training
- All models trained on **Kaggle P100 GPU**
- Classification: Adam/AdamW optimizer, CosineAnnealingLR, early stopping (patience=10)
- Detection: YOLOv8m trained for 86 epochs (early stopping at patience=20), total time: 1.18 hours

### 6. Model Evaluation
- Confusion matrices and classification reports for all classifiers
- ROC curves with AUC scores
- Training loss/accuracy curves
- Detection: mAP@50, per-class AP, F1-confidence curves, PR curves

### 7. Model Comparison
- Unified comparison across all 5 models (`reports/model_comparison.csv`)
- Best classifier: **MobileNetV2** (F1=0.986, fastest at 63ms, smallest at 2.2M params)
- Best detector: **YOLOv8m** (mAP@50=0.813, Drone AP=0.908)

### 8. Streamlit Deployment
- Dual-mode UI: Classification (4-model selector) and Detection (confidence slider)
- `@st.cache_resource` for model caching — loads once, serves instantly
- Interactive confidence bar charts and detection summary tables

### 9. Docker Containerization
- CPU-only PyTorch build (~2.5 GB image vs ~5 GB+ with CUDA)
- Health check endpoint: `/_stcore/health`
- Production-ready with `docker-compose.yml`

---

## 🖥️ Streamlit App

The app offers two modes accessible from the sidebar:

**🐦 Classification Mode**
- Upload an aerial image
- Select any of the 4 trained classifiers from a dropdown (MobileNetV2 recommended)
- View the predicted class (Bird/Drone), confidence score, and probability bar chart

**🎯 Detection Mode**
- Upload an aerial image
- Adjust the confidence threshold slider (0.1–1.0)
- View the annotated image with bounding boxes
- See a detection summary table with class, confidence, and coordinates

---

## 🧪 Testing

Run the pre-deployment verification suite:

```bash
python test_all.py
```

This runs **65 automated tests** covering:
- Essential file existence and model weight integrity
- All module imports
- Classification model loading & inference (all 4 models)
- YOLOv8m detection loading & inference (synthetic + real images)
- Streamlit app syntax validation
- `data.yaml` integrity
- Dockerfile and docker-compose.yml configuration
- `.dockerignore` whitelist verification

Run the model evaluation pipeline:

```bash
python -m src.evaluate                    # All models
python -m src.evaluate --detection-only   # Detection only
python -m src.evaluate --skip-detection   # Classification only
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.12 |
| **Deep Learning** | PyTorch 2.2.2, torchvision 0.17.2 |
| **Object Detection** | Ultralytics YOLOv8 (v8.4.37) |
| **Web UI** | Streamlit 1.56.0 |
| **Metrics** | scikit-learn, seaborn, matplotlib |
| **Augmentation** | torchvision transforms, albumentations |
| **Containerization** | Docker (CPU-only PyTorch) |
| **Cloud Deployment** | Render |
| **GPU Training** | Kaggle P100 |
| **Version Control** | Git, GitHub |

---

## 📌 Skills Demonstrated

- Deep Learning & Computer Vision
- Image Classification (Binary: Bird vs Drone)
- Object Detection with YOLOv8
- Custom CNN Architecture Design
- Transfer Learning (ResNet50, MobileNetV2, EfficientNet-B0)
- Data Preprocessing & Augmentation
- Model Evaluation & Comparison
- Streamlit Web Application Development
- Docker Containerization
- Cloud Deployment (Render)
- End-to-End ML Pipeline Development

---

## 👤 Author

**Aghason Emmanuel Ibeabuchi** (Emmi Dev Codes)
- Location: Lagos, Nigeria
- GitHub: [@Github-Emmi](https://github.com/Github-Emmi)

---

## 📄 License

This project uses the [Roboflow dataset](https://universe.roboflow.com/) — refer to `README.roboflow.txt` for dataset license terms.
