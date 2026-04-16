# 🎬 Video Script — Aerial Object Classification & Detection

**Presenter:** Aghason Emmanuel  
**Total Duration:** ~12 minutes (fits the 12-min project explanation slot)  
**Format:** Screen-recording with voiceover narration

---

## SECTION 1 — Self Introduction *(~1 min)*

> **[SCREEN: GitHub profile or project landing page]**

"Hello, my name is **Aghason Emmanuel**. I am a Data Science and Machine Learning practitioner based in Lagos, Nigeria. Today I'll be walking you through my project — **Aerial Object Classification & Detection** — an end-to-end deep learning pipeline built with PyTorch, YOLOv8, and deployed using Streamlit and Docker."

---

## SECTION 2 — Domain Introduction *(~1 min)*

> **[SCREEN: Show the README.md → Problem Statement section]**

"Aerial surveillance is a rapidly growing field used in **security and defense**, **wildlife protection**, and **airport safety**. Distinguishing between birds and drones in aerial imagery is a critical challenge — a misclassified drone in restricted airspace is a security threat, while a misidentified bird near a wind farm affects wildlife conservation efforts. This project addresses that challenge using deep learning."

---

## SECTION 3 — Problem Statement & Objective *(~1 min)*

> **[SCREEN: Scroll to Problem Statement in README.md]**

"The objective is to build a deep learning solution that can **classify** aerial images as Bird or Drone, and also **detect and localize** these objects in real-world scenes using bounding boxes. The final solution is deployed as an interactive web application for real-time inference."

"In one line: **Given an aerial image, tell me what's in it and where it is.**"

---

## SECTION 4 — Dataset Overview *(~1.5 min)*

> **[SCREEN: Show project root → `classification_dataset/` folder, then `train/`, `valid/`, `test/` folders with bird/drone subfolders]**

"We work with two datasets. First, the **Classification Dataset** — 3,319 RGB JPEG images organized into train, validation, and test splits:
- Train: 1,414 bird + 1,248 drone images
- Validation: 217 bird + 225 drone
- Test: 121 bird + 94 drone

Images are resized to 224×224 pixels and normalized using ImageNet mean and standard deviation."

> **[SCREEN: Show `train/images/` and `train/labels/` folders, open a `.txt` label file]**

"Second, the **Object Detection Dataset** in YOLOv8 format — 3,400 images with bounding box annotations. Each label file contains rows of `class_id x_center y_center width height`. Data split: 2,728 train, 448 validation, 224 test. There's a class imbalance of approximately 1.9:1 — nearly twice as many bird annotations as drone." 

> **[SCREEN: Show `data.yaml`]**

"The `data.yaml` file configures the YOLO dataset — it maps the paths to train, validation, and test image folders, defines two classes: Bird and Drone, and was originally sourced from Roboflow."

---

## SECTION 5 — Data Preprocessing & Augmentation *(~1 min)*

> **[SCREEN: Open `src/preprocessing.py`]**

"All preprocessing is handled in `src/preprocessing.py`. For classification, I normalize pixel values using ImageNet statistics — mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Images are resized to 224×224."

"For **data augmentation** during training, I apply: random crop, horizontal flip, random rotation of ±15 degrees, and color jitter on brightness, contrast, and saturation. Validation and test sets only get resize and normalization — no augmentation — to keep evaluation fair."

"I chose these augmentation techniques because they simulate real-world aerial image variability — lighting changes, camera angles, and orientation differences — without distorting the core object features."

> **[SCREEN: Show `src/data_validation.py` briefly]**

"I also implemented a **data validation module** that checks for image-label pairing, corrupt files, label format correctness, empty labels, class distribution, and cross-split duplicate detection via MD5 hashing — to prevent data leakage."

---

## SECTION 6 — Model Building *(~2.5 min)*

> **[SCREEN: Open `src/models/custom_cnn.py`]**

### Custom CNN

"The first model is a **Custom CNN** I designed from scratch. It's a 4-block architecture: each block has a Convolution layer, Batch Normalization, ReLU activation, Max Pooling, and Dropout2D at 25%. The feature maps go from 3 channels up to 256. The classifier head uses Adaptive Average Pooling, followed by fully connected layers — 256 to 128 to 2 classes — with 50% Dropout to prevent overfitting. It has only **422K parameters** — very lightweight."

"I built this as a **baseline** to establish how well a simple architecture performs before applying transfer learning."

> **[SCREEN: Open `src/models/transfer_learning.py`]**

### Transfer Learning Models

"Next, I used **Transfer Learning** with three ImageNet-pretrained backbones:

1. **ResNet50** — a deep 50-layer residual network. I replaced the final fully-connected layer with Dropout + Linear(2).
2. **MobileNetV2** — a lightweight architecture designed for mobile and edge deployment. Only 2.2 million parameters. Same head replacement.
3. **EfficientNet-B0** — uses compound scaling to balance depth, width, and resolution efficiently.

For all three, I froze 80% of the backbone parameters during initial training and fine-tuned only the top layers and the new classification head. I chose these three specifically to compare a **heavy model** (ResNet50), a **lightweight model** (MobileNetV2), and an **efficient model** (EfficientNet-B0)."

> **[SCREEN: Open `src/models/yolo_detector.py`]**

### YOLOv8m Object Detection

"For detection, I used **YOLOv8m** — the medium variant from Ultralytics. It's an anchor-free, single-stage detector with 25.8 million parameters. I chose YOLOv8m because it balances speed and accuracy well for real-time detection. Training used the `data.yaml` config at image size 416×416."

---

## SECTION 7 — Model Training *(~1 min)*

> **[SCREEN: Show `src/config.py` → ClassificationConfig and DetectionConfig]**

"All models were trained on **Kaggle P100 GPU**. The training configuration is centralized in `src/config.py`:
- **Classification**: Adam/AdamW optimizer, CosineAnnealingLR scheduler, learning rate 1e-3 for custom CNN and 1e-4 for transfer learning, early stopping with patience 10, batch size 32, up to 50 epochs.
- **Detection**: YOLOv8m trained for 86 epochs before early stopping triggered at patience 20. Total training time: 1.18 hours.

I used early stopping and learning rate scheduling because they prevent overfitting and ensure the model converges to the best generalization point."

---

## SECTION 8 — Model Evaluation & Comparison *(~2 min)*

> **[SCREEN: Open `reports/model_comparison.csv`]**

"Here's the unified comparison across all five models:

| Model | Accuracy | F1 Score | ROC-AUC | Parameters | Inference Time |
|-------|----------|----------|---------|------------|----------------|
| Custom CNN | 84.2% | 0.840 | 0.920 | 422K | 58 ms |
| ResNet50 | 97.7% | 0.977 | 0.999 | 23.5M | 175 ms |
| **MobileNetV2** | **98.6%** | **0.986** | **0.999** | **2.2M** | **63 ms** |
| EfficientNet-B0 | 98.6% | 0.986 | 1.000 | 4.0M | 68 ms |
| YOLOv8m | 81.3% mAP@50 | — | — | 25.8M | 5.3 ms |"

> **[SCREEN: Open confusion matrices from `reports/confusion_matrices/`]**

"Looking at the confusion matrices — the Custom CNN misclassifies some drones as birds, which is expected for a simple baseline. The transfer learning models have near-perfect matrices. MobileNetV2 achieves 98% precision on Bird and 100% on Drone."

> **[SCREEN: Open ROC curves from `reports/roc_curves/`]**

"The ROC curves confirm this — all transfer learning models have AUC of 0.999 or higher, meaning nearly perfect class separation."

> **[SCREEN: Open training curves from `reports/training_curves/`]**

"The training curves show clean convergence without overfitting — validation accuracy tracks training accuracy closely, thanks to the augmentation and regularization techniques."

"**Final model selection**: I chose **MobileNetV2** as the best classifier because it ties with EfficientNet-B0 on accuracy and F1, but has **half the parameters** (2.2M vs 4.0M) and **faster inference** (63ms vs 68ms) — making it ideal for deployment."

> **[SCREEN: Show `reports/yolov8m_results.txt`]**

"For detection, YOLOv8m achieved **81.3% mAP@50** on the test set. Drone detection is strong at 90.8% AP, while bird detection is 71.8% AP — birds are harder to detect due to their smaller size and more varied poses. Inference speed is just **5.3 ms per image**."

---

## SECTION 9 — Deployment *(~2 min)*

> **[SCREEN: Show `app/app.py`]**

### Streamlit Application

"The web application is built with **Streamlit**. The entry point is `app/app.py`. The sidebar lets users switch between **Classification mode** and **Detection mode**, and shows the GPU server status."

> **[SCREEN: Open `app/components/classifier_ui.py`]**

"In Classification mode — users upload an image, select from 4 models via a dropdown, and get the prediction with confidence scores and a probability bar chart. In Detection mode — users adjust a confidence threshold slider and see the annotated image with bounding boxes and a detection summary table."

> **[SCREEN: Show `Dockerfile`]**

### Docker & Cloud Deployment

"The app is containerized with Docker using **CPU-only PyTorch** to keep the image small at ~2.5 GB. It's deployed on **Render** at `aerialobjectdetection.onrender.com`."

> **[SCREEN: Show the architecture diagram from README]**

### Hybrid GPU Architecture

"However, Render's free tier has only 512 MB RAM and shared CPU — leading to slow inference and 502 errors. So I designed a **Hybrid Thin Client Architecture**:

```
User Browser → Render (Streamlit UI) → Ngrok Tunnel → Kaggle GPU (FastAPI + PyTorch/YOLO)
```

Render serves just the UI. All inference happens on a **Kaggle T4 GPU** running a FastAPI server exposed via Ngrok. This gives us 10–100x faster inference with zero 502 errors."

> **[SCREEN: Show `kaggle/deploy_inference_server.ipynb` briefly — Cell 3 (server) and Cell 6 (watchdog)]**

"The Kaggle notebook includes a **watchdog** that monitors server health every 30 seconds and auto-restarts uvicorn if it crashes — the Ngrok tunnel URL stays the same. The app auto-detects GPU availability and falls back to local CPU if the server is offline."

---

## SECTION 10 — Project Structure & Code Quality *(~1 min)*

> **[SCREEN: Show project root directory tree in VS Code sidebar]**

"The project follows a **modular structure**:
- `src/` — core source code: config, preprocessing, models, evaluation, validation, utilities
- `app/` — Streamlit UI with separate classifier and detector components
- `kaggle/` — GPU training scripts and the inference server notebook
- `models/` — saved model weights (classification + detection)
- `reports/` — all evaluation artifacts: confusion matrices, ROC curves, training curves, model comparison CSV
- `scripts/` — utility scripts for dataset validation and Kaggle integration
- `test_all.py` — **65 automated tests** covering file integrity, model loading, inference, Docker config, and more

Every module is documented, the code is reusable, and the configuration is centralized in `src/config.py`."

---

## SECTION 11 — Live Demo *(~1 min)*

> **[SCREEN: Open the Streamlit app at https://aerialobjectdetection.onrender.com]**

"Let me show the app in action."

> **[ACTION: Upload a bird image → select MobileNetV2 → click Classify]**

"I upload an aerial image, select MobileNetV2 from the dropdown, and within milliseconds we get: **Bird — 98% confidence**. The bar chart shows the probability distribution."

> **[ACTION: Switch to Detection mode → upload an image → set threshold to 0.25 → run]**

"Switching to Detection mode — I upload an image, set the confidence threshold to 0.25, and YOLOv8m draws bounding boxes around detected objects with class labels and confidence scores. Below, we see the detection summary table."

---

## SECTION 12 — Conclusion & Business Suggestions *(~30 sec)*

> **[SCREEN: Show the Key Results table in README.md]**

"To summarize:
- **MobileNetV2** is the best classifier — 98.6% accuracy, 0.986 F1 score, with only 2.2M parameters
- **YOLOv8m** achieves 81.3% mAP@50 for real-time object detection
- The full pipeline — from data validation to cloud deployment with GPU inference — is **production-ready**

**Business suggestion:** This solution can be deployed at airports for bird-strike prevention, at borders for unauthorized drone detection, or at wind farms for wildlife monitoring — all requiring only a camera feed and this inference engine."

"Thank you."

---

## 📝 Screen Recording Checklist

Use this to plan your recording flow:

1. [ ] Open GitHub profile or project repo page → Self Introduction
2. [ ] Open `README.md` → Scroll through Problem Statement, Datasets, Key Results
3. [ ] Show `classification_dataset/` folder structure in VS Code
4. [ ] Show `train/labels/` → open a `.txt` annotation file
5. [ ] Open `data.yaml`
6. [ ] Open `src/preprocessing.py` → show transforms and augmentation
7. [ ] Open `src/data_validation.py` briefly
8. [ ] Open `src/models/custom_cnn.py` → walk through architecture
9. [ ] Open `src/models/transfer_learning.py` → show 3 backbones
10. [ ] Open `src/models/yolo_detector.py` → show YOLO wrapper
11. [ ] Open `src/config.py` → show training hyperparameters
12. [ ] Open `reports/model_comparison.csv` → show metrics table
13. [ ] Open confusion matrix PNGs from `reports/confusion_matrices/`
14. [ ] Open ROC curve PNGs from `reports/roc_curves/`
15. [ ] Open training curve PNGs from `reports/training_curves/`
16. [ ] Open `reports/yolov8m_results.txt` → show detection metrics
17. [ ] Open `app/app.py` → show entry point
18. [ ] Open `app/components/classifier_ui.py` → show UI logic
19. [ ] Open `Dockerfile` → show containerization
20. [ ] Show architecture diagram (README hybrid section)
21. [ ] Briefly show `kaggle/deploy_inference_server.ipynb`
22. [ ] Show VS Code sidebar → full project tree
23. [ ] Open Streamlit app → Demo Classification
24. [ ] Demo Detection
25. [ ] Show Key Results table → Conclusion
