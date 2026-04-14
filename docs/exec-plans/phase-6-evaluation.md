# ExecPlan: Phase 6 — Model Evaluation & Comparison Report

**Created**: 2026-04-13
**Last Updated**: 2026-04-13
**Status**: ✅ Complete

---

## Why This Matters

After this work, a user can run a single command and see a complete comparison of all 5 trained models (Custom CNN, ResNet50, MobileNetV2, EfficientNet-B0, YOLOv8m) with accuracy, precision, recall, F1-score, confusion matrices, and per-class AP — all rendered as terminal output, saved CSV reports, and matplotlib plots. This is the final judgment of which model to deploy in the Streamlit app (Phase 7).

---

## Prerequisites

- **Python**: 3.12 (pyenv-managed venv at `venv/`)
- **OS**: macOS (Intel, no local GPU — evaluation runs on CPU with downloaded weights)
- **Working directory**: `/Volumes/EmmiDev256G/Projects/object_detection_Dataset`
- **Prior phases completed**:
  - Phase 1: Data Validation ✅ (3,400 detection images, 3,319 classification, 81 empty labels)
  - Phase 2: Preprocessing ✅ (`src/preprocessing.py` with dataloaders)
  - Phases 3-5: Model architectures + training code ✅
  - Kaggle GPU training: Classification kernel v4 running, detection kernel not yet pushed
- **Kaggle CLI**: Authenticated (`KAGGLE_API_TOKEN` set in `venv/bin/activate`)
- **Kaggle username**: `aghasonemmanuel`
- **Kaggle dataset slug**: `aghasonemmanuel/aerial-bird-drone-detection`
- **Kaggle classification kernel slug**: `aghasonemmanuel/aerial-classification-training`

### What Already Exists

| Path | What It Does | Status |
|------|-------------|--------|
| `src/config.py` | Central hyperparameters (ClassificationConfig, DetectionConfig) | ✅ Exists |
| `src/preprocessing.py` | Classification dataloaders (224×224, ImageNet norm) | ✅ Exists |
| `src/models/custom_cnn.py` | AerialCNN — 4-block CNN, ~422K params | ✅ Exists |
| `src/models/transfer_learning.py` | ResNet50/MobileNetV2/EfficientNet-B0, 80% frozen | ✅ Exists |
| `src/models/yolo_detector.py` | YOLOv8m training wrapper | ✅ Exists |
| `src/train_classifier.py` | Classification training loop with early stopping + TensorBoard | ✅ Exists |
| `src/train_detector.py` | YOLOv8m training script | ✅ Exists |
| `kaggle/train_classification_kaggle.py` | Self-contained Kaggle GPU classification script (v4) | ✅ Exists |
| `kaggle/train_detection_kaggle.py` | Self-contained Kaggle GPU detection script | ✅ Exists |
| `kaggle/kernel-metadata.json` | Kernel config (currently pointing to classification script) | ✅ Exists |
| `scripts/kaggle_pull_outputs.sh` | Download trained weights from Kaggle | ✅ Exists |
| `src/evaluate.py` | **DOES NOT EXIST** — this is the main deliverable | ❌ Create |
| `src/utils.py` | **DOES NOT EXIST** — visualization helpers | ❌ Create |
| `models/classification/` | Trained `.pth` weights — empty, need to pull from Kaggle | ❌ Empty |
| `models/detection/` | Trained `best.pt` weights — empty, need to pull from Kaggle | ❌ Empty |

### Dataset Quick Reference (verified ground truth)

| Dataset | Images | Classes | Format |
|---------|--------|---------|--------|
| Detection | 3,400 (train 2,728 / valid 448 / test 224) | Bird (0), drone (1) | YOLOv8 `.txt` labels |
| Classification | 3,319 (train 2,662 / valid 442 / test 215) | bird/, drone/ folders | `ImageFolder` |
| Bbox counts | Bird: 3,157 / Drone: 1,702 (1.9:1 ratio) | — | — |
| Empty labels | 81 (train 66 / valid 6 / test 9) | Background/hard negatives | — |

> **NOTE**: The bbox counts were re-verified during Phase 1 validation (Bird **3,157** : Drone **1,702** = 1.9:1). The `references/project-context.md` file still shows the old numbers (1,406:135 = 10.4:1). The corrected numbers in `aerial-detection.agent.md` (top section) are authoritative.

---

## Repository Orientation

After completing this ExecPlan, the new/modified files will be:

```
object_detection_Dataset/
├── src/
│   ├── evaluate.py              # NEW: Evaluation metrics + comparison report
│   └── utils.py                 # NEW: Visualization helpers (confusion matrix plots, etc.)
├── models/
│   ├── classification/          # POPULATED: Downloaded from Kaggle
│   │   ├── custom_cnn/
│   │   │   ├── best_model.pth
│   │   │   ├── training_log.csv
│   │   │   └── test_results.txt
│   │   ├── resnet50/
│   │   │   └── (same structure)
│   │   ├── mobilenet_v2/
│   │   │   └── (same structure)
│   │   └── efficientnet_b0/
│   │       └── (same structure)
│   └── detection/               # POPULATED: Downloaded from Kaggle
│       ├── best.pt
│       └── test_results.txt
├── reports/                     # NEW: Evaluation output directory
│   ├── model_comparison.csv
│   ├── classification_report.txt
│   ├── confusion_matrices/      # Per-model confusion matrix plots
│   └── training_curves/         # Per-model loss/accuracy plots
└── kaggle/
    └── kernel-metadata.json     # MODIFIED: Switched to detection before pushing
```

**How these files connect**:
1. `src/evaluate.py` loads `.pth` weights (classification) and `best.pt` (detection) from `models/`
2. It runs inference on the test splits of both datasets
3. It computes metrics (accuracy, precision, recall, F1, confusion matrix, ROC-AUC for classification; mAP@0.5, mAP@0.5:0.95, per-class AP for detection)
4. It writes results to `reports/` as CSV tables, text summaries, and matplotlib plots
5. `src/utils.py` provides shared plotting functions used by `evaluate.py`

---

## Milestones

### Milestone 1: Pull Classification Training Weights from Kaggle

**Status**: ⬜ Not Started

**Context**: Kernel v4 was pushed with a fixed GPU compatibility shim (uses `nvidia-smi` to detect P100, explicitly uninstalls old torch/torchvision packages to avoid stale files, then installs `torch==2.2.2+cu118`). Previous v1-v3 failures were caused by: (v1) `total_mem` typo + P100 incompatibility, (v2) `torch==2.1.2` not found on cu118 index, (v3) old `torchvision._meta_registrations` files persisting after pip install.

**What to do**:

1. Check kernel v4 status:
   ```bash
   cd /Volumes/EmmiDev256G/Projects/object_detection_Dataset
   source venv/bin/activate
   kaggle kernels status aghasonemmanuel/aerial-classification-training
   ```
   Wait until status is `complete`. If status is `error`, pull logs:
   ```bash
   kaggle kernels output aghasonemmanuel/aerial-classification-training -p /tmp/kaggle-debug-v4
   cat /tmp/kaggle-debug-v4/aerial-classification-training.log
   ```

2. Once `complete`, download outputs:
   ```bash
   bash scripts/kaggle_pull_outputs.sh classification
   ```
   This downloads to `models/classification/`.

3. Verify all 4 model weights exist:
   ```bash
   for m in custom_cnn resnet50 mobilenet_v2 efficientnet_b0; do
     echo "--- $m ---"
     ls -lh models/classification/$m/best_model.pth 2>/dev/null || echo "MISSING"
     cat models/classification/$m/test_results.txt 2>/dev/null || echo "NO RESULTS"
   done
   ```

**Verification**:
```bash
python3 -c "
import torch
for m in ['custom_cnn', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']:
    path = f'models/classification/{m}/best_model.pth'
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
        print(f'✓ {m}: {len(state)} layers loaded')
    except FileNotFoundError:
        print(f'✗ {m}: MISSING')
"
```

**Expected output**:
```
✓ custom_cnn: 24 layers loaded
✓ resnet50: ~318 layers loaded
✓ mobilenet_v2: ~158 layers loaded
✓ efficientnet_b0: ~213 layers loaded
```

**If it fails**:
- `error` status → Check log for Python errors. The v4 fix addresses the known P100/torchvision issue. If a new error appears, read the log, fix the script, and push v5.
- `running` for >45 min → Training 4 models × 50 epochs on P100 can take 30-45 min. Wait longer.
- Missing model subdirectories → The Kaggle script saves to `/kaggle/working/classification/{model_name}/`. Check log for path discovery output.

---

### Milestone 2: Push and Pull Detection Training from Kaggle

**Status**: ⬜ Not Started

**What to do**:

1. Switch `kaggle/kernel-metadata.json` to point to the detection script:
   ```bash
   cd /Volumes/EmmiDev256G/Projects/object_detection_Dataset
   ```
   Edit `kaggle/kernel-metadata.json`:
   - Change `"id"` to `"aghasonemmanuel/aerial-detection-training"`
   - Change `"title"` to `"Aerial Detection Training"`
   - Change `"code_file"` to `"train_detection_kaggle.py"`

2. Push the detection kernel:
   ```bash
   kaggle kernels push -p kaggle/
   ```

3. Monitor until complete:
   ```bash
   kaggle kernels status aghasonemmanuel/aerial-detection-training
   ```
   YOLOv8m training (100 epochs, 416×416, batch 16) takes ~60-90 min on P100.

4. Pull outputs:
   ```bash
   mkdir -p models/detection
   kaggle kernels output aghasonemmanuel/aerial-detection-training -p models/detection
   ```

**Verification**:
```bash
ls -lh models/detection/best.pt && echo "✓ YOLOv8m weights downloaded"
cat models/detection/test_results.txt
```

**Expected output**:
```
-rw-r--r--  1 user  staff  ~52M  date time  models/detection/best.pt
✓ YOLOv8m weights downloaded
model=yolov8m
test_mAP50=0.XXXX
test_mAP50_95=0.XXXX
```

**If it fails**:
- Detection uses `ultralytics` + `yaml` imports. Verify `train_detection_kaggle.py` has the same `ensure_gpu_compatible_pytorch()` fix as v4 of the classification script.
- Path issues: The detection script expects `{data_root}/train/images/` and `{data_root}/valid/images/` on Kaggle. These are in the uploaded dataset's tar structure.

---

### Milestone 3: Create `src/evaluate.py` — Classification Evaluation

**Status**: ⬜ Not Started

**What to create**: `src/evaluate.py` — a script that loads trained classification model weights, runs inference on the test set, and computes:

- **Accuracy** (overall correct / total)
- **Precision** (per-class and weighted average)
- **Recall** (per-class and weighted average)
- **F1-Score** (per-class and weighted average)
- **Confusion Matrix** (2×2: Bird vs drone)
- **ROC-AUC** (using softmax probabilities)
- **Inference time** (average ms per image on CPU)

**Key design decisions**:
- Load weights from `models/classification/{model_name}/best_model.pth`
- Use the same dataloaders from `src/preprocessing.py` (test split only)
- Use `sklearn.metrics` for classification_report, confusion_matrix, roc_auc_score
- Output to `reports/` directory as CSV + text + plots
- Must handle all 4 classification models in a single run

**Implementation**:
```python
# src/evaluate.py
"""Model evaluation and comparison report generation.

Usage:
    python -m src.evaluate                        # Evaluate all models
    python -m src.evaluate --models custom_cnn    # Evaluate specific model
    python -m src.evaluate --detection-only       # YOLOv8m only
"""
```

The script should:
1. Accept `--models` arg (list of classification model names) and `--detection-only` flag
2. Load each model, run on test set, compute all metrics
3. Save per-model results to `reports/{model_name}_results.txt`
4. Save confusion matrix plots to `reports/confusion_matrices/{model_name}.png`
5. Generate a comparison table (`reports/model_comparison.csv`) with all models side by side
6. Print the comparison table to stdout

**Verification**:
```bash
python -m src.evaluate --models custom_cnn
```

**Expected output** (approximate — depends on training results):
```
Loading custom_cnn from models/classification/custom_cnn/best_model.pth
Running inference on 215 test images...

Custom CNN Results:
  Accuracy:  0.XXXX
  Precision: 0.XXXX (weighted)
  Recall:    0.XXXX (weighted)
  F1-Score:  0.XXXX (weighted)
  ROC-AUC:   0.XXXX

Confusion Matrix:
         Predicted
         Bird  drone
Bird     [XX]  [XX]
drone    [XX]  [XX]

Saved: reports/confusion_matrices/custom_cnn.png
Saved: reports/custom_cnn_results.txt
```

**If it fails**:
- `FileNotFoundError` for `.pth` file → Milestone 1 must complete first
- Mismatch in model architecture vs weights → Ensure `build_model()` function matches the architecture used in the Kaggle training script (same AerialCNN class, same transfer learning config)

---

### Milestone 4: Create `src/evaluate.py` — Detection Evaluation (Per-Class AP)

**Status**: ⬜ Not Started

**What to add**: Extend `src/evaluate.py` with YOLOv8m detection evaluation. This is critical because agent.md requires **per-class AP reporting** (not just mAP) to ensure the 1.9:1 Bird:Drone bbox imbalance isn't hiding poor drone detection.

**Detection metrics to compute**:
- **mAP@0.5** (mean Average Precision at IoU threshold 0.5)
- **mAP@0.5:0.95** (averaged across IoU thresholds 0.5 to 0.95 in 0.05 steps)
- **Per-class AP@0.5**: Bird AP and Drone AP separately
- **Precision** (per-class)
- **Recall** (per-class)
- **Background performance**: Verify 81 empty-label images produce zero false positives

**Implementation**:
```python
# Detection evaluation uses ultralytics YOLO.val()
from ultralytics import YOLO

def evaluate_detection(weights_path, data_yaml_path, ...):
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml_path, split="test", ...)
    # Extract per-class AP
    bird_ap50 = metrics.box.ap50[0]
    drone_ap50 = metrics.box.ap50[1]
    ...
```

**Verification**:
```bash
python -m src.evaluate --detection-only
```

**Expected output**:
```
Loading YOLOv8m from models/detection/best.pt
Running detection evaluation on 224 test images...

YOLOv8m Detection Results:
  mAP@0.5:     0.XXXX
  mAP@0.5:0.95: 0.XXXX
  Bird AP@0.5:  0.XXXX
  Drone AP@0.5: 0.XXXX
  Bird Precision: 0.XXXX  Recall: 0.XXXX
  Drone Precision: 0.XXXX  Recall: 0.XXXX

Saved: reports/detection_results.txt
```

---

### Milestone 5: Create `src/utils.py` — Visualization Helpers

**Status**: ⬜ Not Started

**What to create**: `src/utils.py` with reusable plotting functions:

1. `plot_confusion_matrix(y_true, y_pred, class_names, save_path)` — Seaborn heatmap
2. `plot_training_curves(csv_path, save_path)` — Loss + accuracy over epochs from training_log.csv
3. `plot_roc_curve(y_true, y_probs, save_path)` — ROC curve with AUC annotation
4. `plot_model_comparison(comparison_df, metric, save_path)` — Bar chart of all models

Dependencies: `matplotlib`, `seaborn`, `scikit-learn` (all in `requirements.txt`).

**Verification**:
```bash
python -c "from src.utils import plot_confusion_matrix; print('✓ utils imported')"
```

---

### Milestone 6: Generate Full Comparison Report

**Status**: ⬜ Not Started

**What to do**: Run the complete evaluation pipeline and produce the comparison table from agent.md Phase 6:

```bash
python -m src.evaluate
```

This should produce `reports/model_comparison.csv` matching this format:

| Model | Accuracy/mAP | Precision | Recall | F1 | Params | Inference (ms) |
|-------|-------------|-----------|--------|-----|--------|---------------|
| Custom CNN | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 422K | XX.X |
| ResNet50 | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 23.5M | XX.X |
| MobileNetV2 | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 2.2M | XX.X |
| EfficientNet-B0 | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 4.0M | XX.X |
| YOLOv8m | 0.XXXX (mAP50) | 0.XXXX | 0.XXXX | — | 25.9M | XX.X |

**Selection criteria** (from agent.md):
- Best classification model: highest **F1-score** (accounts for class imbalance)
- Detection model: highest **mAP@0.5**
- The winning classification model + YOLOv8m go into the Streamlit app (Phase 7)

**Verification**:
```bash
cat reports/model_comparison.csv
ls reports/confusion_matrices/*.png | wc -l   # Should show 4 (one per classifier)
ls reports/training_curves/*.png | wc -l      # Should show 4
```

**Expected output**:
```
model,accuracy,precision,recall,f1,roc_auc,params,inference_ms
custom_cnn,0.XXXX,0.XXXX,0.XXXX,0.XXXX,0.XXXX,422178,XX.X
resnet50,0.XXXX,0.XXXX,...
...
4  (confusion matrices)
4  (training curves)
```

---

## Progress

| Date | Milestone | Status | Notes |
|------|-----------|--------|-------|
| 2026-04-13 | Plan created | ✅ Complete | ExecPlan authored with 6 milestones |
| 2026-04-13 | Milestone 1: Pull classification weights | ✅ Complete | Kaggle v5 succeeded, all 4 models verified |
| 2026-04-13 | Milestone 2: Push+pull detection training | ✅ Complete | v4 succeeded after 3 fixes (path, ultralytics, numpy order). 86 epochs, 1.18h on P100 |
| 2026-04-13 | Milestone 3: Classification eval code | ✅ Complete | `src/evaluate.py` with CLI, all 4 classifiers evaluated |
| 2026-04-13 | Milestone 4: Detection eval code | ✅ Complete | Per-class AP: Bird 0.718, Drone 0.908 (test set) |
| 2026-04-13 | Milestone 5: Visualization helpers | ✅ Complete | `src/utils.py` — confusion matrix, ROC, training curves, comparison charts |
| 2026-04-13 | Milestone 6: Full comparison report | ✅ Complete | 5-model comparison CSV, all plots saved |

---

## Decision Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-04-13 | Evaluate on **CPU** (local Mac) not Kaggle GPU | Evaluation doesn't need GPU — inference on 215 test images is fast on CPU; avoids burning Kaggle GPU quota |
| 2026-04-13 | Use `sklearn.metrics` for classification metrics | Standard library for precision/recall/F1/confusion matrix; already in `requirements.txt` |
| 2026-04-13 | Use `YOLO.val()` for detection metrics instead of custom | Ultralytics computes mAP correctly; avoids re-implementing IoU thresholds, NMS, AP calculation |
| 2026-04-13 | Select best classifier by **F1-score** not accuracy | Agent.md explicitly says F1 (accounts for class imbalance); accuracy alone can be misleading |
| 2026-04-13 | Save reports to `reports/` not `models/` | Separation of concerns — `models/` has weights only; `reports/` has all evaluation artifacts |

---

## Surprises & Discoveries

### 2026-04-13: Bbox Count Discrepancy in Documentation

**What happened**: The `references/project-context.md` and agent.md Phase 3 section still reference the **old** bbox counts (Bird 1,406 : Drone 135 = 10.4:1 ratio). The **corrected** counts from Phase 1 validation are Bird **3,157** : Drone **1,702** = **1.9:1** ratio. The agent.md top-level bbox table was updated but Phase 3 was not.

**Impact on plan**: The 1.9:1 ratio is moderate — no aggressive class weighting is needed. YOLOv8's built-in focal loss (`fl_gamma=1.5`) is sufficient. The evaluation should still report per-class AP to confirm drone detection quality.

### 2026-04-13: Kaggle Kernel v1-v3 Failure History

**What happened**:
- v1: `AttributeError: total_mem` (should be `total_memory`) + P100 sm_60 incompatible with pre-installed PyTorch 2.10.0+cu128
- v2: `pip install torch==2.1.2` failed (not available on cu118 index; minimum is 2.2.0)
- v3: After installing `torch==2.2.2+cu118`, old `torchvision 2.10.0` files persisted in `dist-packages`, causing `RuntimeError: operator torchvision::nms does not exist`
- v4 fix: Uses `nvidia-smi -L` to detect P100 (no torch import), explicitly `pip uninstall` + `shutil.rmtree` to clean old packages, then installs fresh `torch==2.2.2+cu118` + `torchvision==0.17.2`

**Impact on plan**: v4 should resolve the issue. If it doesn't, the fallback is to script the install steps in a separate bash script that runs before the Python training script.

---

## Outcomes & Retrospective

### What Was Achieved
- All 5 models evaluated: 4 classifiers + YOLOv8m detector
- Best classifier: **EfficientNet-B0** (F1=0.9860, ROC-AUC=0.9996) — tied with MobileNetV2 on F1
- Best for deployment: **MobileNetV2** (2.2M params, 63ms/img, same F1 as EfficientNet)
- Detection: **YOLOv8m** mAP@0.5=0.813 (test), mAP@0.5:0.95=0.524
- Detection per-class: Drone AP@0.5=0.908 (strong), Bird AP@0.5=0.718 (acceptable, lower recall)
- Early stopping triggered at epoch 86/100 (patience=20), best ~epoch 66
- Val/test metrics nearly identical — excellent generalization, no overfitting
- Full reports: model_comparison.csv, confusion matrices, ROC curves, training curves, F1 comparison

### What Remains
- Phase 7: Streamlit Deployment (MobileNetV2 or EfficientNet-B0 + YOLOv8m)
- Phase 8: Docker Containerization
- README documentation

### Lessons Learned
- **Kaggle dependency order matters**: numpy<2 MUST be installed LAST (after ultralytics), as ultralytics pulls numpy>=2 which breaks torch 2.2.2
- **Kaggle dataset mount path**: `/kaggle/input/datasets/{username}/{dataset-slug}/` NOT `/kaggle/input/{dataset-slug}/`
- **Smart path discovery**: Use candidate lists + rglob fallback rather than hardcoding Kaggle paths
- **macOS `._` resource forks**: Double image counts (5456 vs 2728) but ultralytics ignores non-image files automatically

### Observable Proof

```bash
# The user can verify the completed evaluation by running:
python -m src.evaluate
cat reports/model_comparison.csv
```
