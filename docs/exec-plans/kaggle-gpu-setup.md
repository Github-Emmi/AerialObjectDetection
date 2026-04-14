# ExecPlan: Kaggle GPU Environment Setup for Aerial Object Classification & Detection

**Created**: 2026-04-13
**Last Updated**: 2026-04-13
**Status**: 🟡 Running — Classification kernel executing on Kaggle GPU

---

## Why This Matters

Training deep learning models (Custom CNN, ResNet50, MobileNetV2, EfficientNet-B0, YOLOv8m) on CPU takes hours to days. After completing this ExecPlan, you will be able to push training jobs from your local VS Code terminal to Kaggle's free GPU (NVIDIA Tesla T4 / P100, 16 GB VRAM), monitor their status, and pull back trained model weights — all without leaving VS Code. No Kaggle web UI interaction is needed after initial one-time setup.

---

## Key Terms

- **Kaggle**: A Google-owned platform that provides free GPU/TPU compute for running Python notebooks and scripts. Free accounts get ~30 hours/week of GPU time.
- **Kaggle Kernel**: A notebook or script that runs on Kaggle's servers. We will use **script** mode (a plain `.py` file) rather than a Jupyter notebook for reproducibility.
- **Kaggle Dataset**: A data bundle uploaded to Kaggle that kernels can mount as read-only input. We upload our bird/drone images once, and every kernel run can access them.
- **`kaggle` CLI**: A command-line tool (`pip install kaggle`) that lets you upload datasets, push kernels, check status, and download outputs — all from a terminal.
- **`kernel-metadata.json`**: A JSON file that tells Kaggle how to configure a kernel run: which script to execute, which datasets to mount, whether to enable GPU, and internet access.
- **`dataset-metadata.json`**: A JSON file that provides the title and slug for a Kaggle dataset upload.
- **Accelerator**: Kaggle's term for GPU/TPU hardware. We will use `NvidiaTeslaT4` (free tier, 16 GB VRAM). Other options include `NvidiaTeslaP100`, `NvidiaL4`, and `NvidiaTeslaA100` (availability varies).

---

## Prerequisites

- **OS**: macOS (this project runs on an Intel Mac; the guide works on Linux/Windows WSL too)
- **Python**: 3.12 (managed via `pyenv`; the local venv at `venv/` already has all packages installed)
- **Working directory**: The repository root — the folder containing `data.yaml`, `src/`, `train/`, `valid/`, `test/`, and `classification_dataset/`
- **Kaggle account**: Free account at https://www.kaggle.com (sign up with Google/email)
- **VS Code**: Any recent version with a terminal
- **Internet**: Required for uploading datasets and pushing kernels
- **Prior phases completed**: Phase 1 (Data Validation) and Phase 2 (Preprocessing & Augmentation) — the `src/` code is already written and locally tested

### What Already Exists in This Repository

| Path | What It Does | Status |
|------|-------------|--------|
| `src/config.py` | Hyperparameters, paths, class names | ✅ Exists |
| `src/preprocessing.py` | Classification dataloaders (224×224, ImageNet norm) | ✅ Exists |
| `src/models/custom_cnn.py` | 4-block CNN: 3→32→64→128→256 channels, ~422K params | ✅ Exists |
| `src/models/transfer_learning.py` | ResNet50/MobileNetV2/EfficientNet-B0 with frozen backbones | ✅ Exists |
| `src/models/yolo_detector.py` | YOLOv8m training wrapper | ✅ Exists |
| `src/train_classifier.py` | Training loop with early stopping + TensorBoard | ✅ Exists |
| `src/train_detector.py` | YOLOv8m training script | ✅ Exists |
| `classification_dataset/{train,valid,test}/{bird,drone}/` | Classification images (3,319 total) | ✅ Exists |
| `{train,valid,test}/{images,labels}/` | Detection dataset (3,400 images, YOLOv8 format) | ✅ Exists |
| `data.yaml` | YOLOv8 dataset config: classes `['Bird', 'drone']` | ✅ Exists |
| `requirements.txt` | All Python dependencies | ✅ Exists |
| `kaggle/` | Kaggle kernel scripts + metadata | 🔨 Create in this plan |

---

## Repository Orientation

After completing this ExecPlan, the new files will be:

```
object_detection_Dataset/
├── kaggle/                                 # NEW: All Kaggle-related files
│   ├── kernel-metadata.json                # Tells Kaggle: GPU=on, which dataset, which script
│   ├── train_classification_kaggle.py      # Self-contained classification training script
│   ├── train_detection_kaggle.py           # Self-contained YOLOv8m training script
│   └── dataset-metadata.json               # Metadata for the uploaded Kaggle dataset
├── scripts/
│   ├── kaggle_push.sh                      # NEW: One-command push + monitor workflow
│   └── kaggle_pull_outputs.sh              # NEW: Download trained model weights
└── ... (existing files unchanged)
```

**How these files connect**:
1. You run `scripts/kaggle_push.sh classification` from VS Code terminal
2. That script calls `kaggle kernels push -p kaggle/` which uploads `train_classification_kaggle.py` to Kaggle
3. Kaggle runs the script on a T4 GPU, reading the dataset you previously uploaded
4. When done, you run `scripts/kaggle_pull_outputs.sh` to download `.pth` weight files + TensorBoard logs back to your local `models/` directory

---

## Milestones

### Milestone 1: Install & Authenticate Kaggle CLI

**Status**: ✅ Complete

**What to do**:

1. Activate your project virtual environment:
   ```bash
   cd /Volumes/EmmiDev256G/Projects/object_detection_Dataset
   source venv/bin/activate
   ```

2. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

3. Get your API credentials:
   - Go to https://www.kaggle.com/settings
   - Under the **"API"** section, click **"Create New Token"**
   - This downloads a file called `kaggle.json` containing your username and API key
   - It looks like: `{"username":"your-kaggle-username","key":"abcdef1234567890abcdef1234567890"}`

4. Place the credentials file:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
   The `chmod 600` restricts read access to only your user — Kaggle CLI warns if permissions are too open.

5. **SECURITY**: Never commit `kaggle.json` to git. It is already covered by `.gitignore` (the `*.json` pattern under `~/.kaggle/` is outside the repo, so it's safe). But as extra protection:
   ```bash
   echo '~/.kaggle/' >> .gitignore   # Not strictly needed since it's outside the repo
   ```

**Verification**:
```bash
kaggle --version
```

**Expected output** (version may differ):
```
Kaggle CLI 2.x.x
```

Then verify authentication:
```bash
kaggle datasets list --mine
```

**Expected output**: An empty table or a list of your datasets (no authentication error).

**If it fails**:
- `"Could not find kaggle.json"` → Check that `~/.kaggle/kaggle.json` exists and has correct content
- `"403 - Forbidden"` → Your API key may be expired; regenerate it at https://www.kaggle.com/settings
- `"Command not found: kaggle"` → Run `pip install kaggle` again in the active venv, or check `which kaggle`

---

### Milestone 2: Upload the Dataset to Kaggle

**Status**: ✅ Complete

We need to upload both the classification and detection datasets to Kaggle so our kernels can access them. We will create a **single Kaggle dataset** containing all images and labels.

**What to do**:

1. Create the dataset metadata file at `kaggle/dataset-metadata.json`:
   ```json
   {
     "title": "Aerial Bird Drone Detection Dataset",
     "id": "YOUR_KAGGLE_USERNAME/aerial-bird-drone-detection",
     "licenses": [{"name": "CC-BY-4.0"}]
   }
   ```
   **IMPORTANT**: Replace `YOUR_KAGGLE_USERNAME` with your actual Kaggle username (e.g., `emmidev/aerial-bird-drone-detection`).

2. Create a staging directory with symlinks (to avoid duplicating ~600 MB of images):
   ```bash
   cd /Volumes/EmmiDev256G/Projects/object_detection_Dataset

   mkdir -p kaggle/dataset-staging
   
   # Copy lightweight files
   cp data.yaml kaggle/dataset-staging/
   cp requirements.txt kaggle/dataset-staging/
   
   # Copy source code
   cp -r src kaggle/dataset-staging/
   
   # Symlink image directories (saves disk space)
   ln -sf "$(pwd)/train" kaggle/dataset-staging/train
   ln -sf "$(pwd)/valid" kaggle/dataset-staging/valid
   ln -sf "$(pwd)/test" kaggle/dataset-staging/test
   ln -sf "$(pwd)/classification_dataset" kaggle/dataset-staging/classification_dataset
   
   # Copy dataset metadata into staging
   cp kaggle/dataset-metadata.json kaggle/dataset-staging/dataset-metadata.json
   ```

3. Upload the dataset to Kaggle:
   ```bash
   kaggle datasets create -p kaggle/dataset-staging -r tar --quiet
   ```
   The `-r tar` flag tells Kaggle to upload subdirectories as uncompressed tar archives (preserving folder structure). This upload may take 10-30 minutes depending on your internet speed (~600 MB of images).

4. Verify the upload:
   ```bash
   kaggle datasets status YOUR_KAGGLE_USERNAME/aerial-bird-drone-detection
   ```

**Verification**:
```bash
kaggle datasets files YOUR_KAGGLE_USERNAME/aerial-bird-drone-detection --page-size 5
```

**Expected output**: A table listing some of the uploaded files (images, labels, data.yaml).

**If it fails**:
- `"403 Forbidden"` → Check your `kaggle.json` credentials
- Upload is slow → This is normal for ~600 MB; let it finish
- `"Dataset already exists"` → Use `kaggle datasets version -p kaggle/dataset-staging -m "re-upload" -r tar` to create a new version

---

### Milestone 3: Create the Classification Training Kernel Script

**Status**: ✅ Complete

This creates a **self-contained Python script** that Kaggle will execute on its GPU. The script includes everything needed — no imports from `src/` (Kaggle cannot access your local repo structure directly, but we uploaded `src/` as part of the dataset).

**What to do**:

Create the file `kaggle/train_classification_kaggle.py` (see the exact content provided in the implementation section below).

Key design decisions for the Kaggle script:
- **Dataset path**: On Kaggle, datasets are mounted at `/kaggle/input/<dataset-slug>/`. Our dataset will be at `/kaggle/input/aerial-bird-drone-detection/`
- **Output path**: Kaggle kernels can write to `/kaggle/working/`. Anything in this directory is downloadable as output after the run
- **GPU detection**: The script detects `cuda` automatically — works on both GPU (Kaggle) and CPU (local)
- **Saves**: Model weights (`.pth`), training logs (CSV), and best accuracy summary are written to `/kaggle/working/`

**Verification**:
```bash
ls -la kaggle/train_classification_kaggle.py
head -5 kaggle/train_classification_kaggle.py
```

**Expected output**: The file exists and starts with `#!/usr/bin/env python3` and a docstring.

---

### Milestone 4: Create the Detection Training Kernel Script

**Status**: ✅ Complete

Same pattern as Milestone 3, but for YOLOv8m object detection training.

**What to do**:

Create the file `kaggle/train_detection_kaggle.py` (see exact content in implementation section below).

Key differences from classification:
- Uses `ultralytics` YOLO API (not raw PyTorch training loop)
- Reads `data.yaml` from the dataset mount path
- Requires a modified `data.yaml` with Kaggle-specific absolute paths
- Saves `best.pt` and `last.pt` weights + metrics to `/kaggle/working/`

**Verification**:
```bash
ls -la kaggle/train_detection_kaggle.py
```

---

### Milestone 5: Create Kernel Metadata and Push/Pull Scripts

**Status**: ✅ Complete

**What to do**:

1. Create `kaggle/kernel-metadata.json`:
   ```json
   {
     "id": "YOUR_KAGGLE_USERNAME/aerial-classification-training",
     "title": "Aerial Classification Training",
     "code_file": "train_classification_kaggle.py",
     "language": "python",
     "kernel_type": "script",
     "is_private": "true",
     "enable_gpu": "true",
     "enable_internet": "true",
     "dataset_sources": ["YOUR_KAGGLE_USERNAME/aerial-bird-drone-detection"],
     "competition_sources": [],
     "kernel_sources": [],
     "model_sources": []
   }
   ```

2. Create `scripts/kaggle_push.sh` — a one-command wrapper that updates the metadata and pushes.

3. Create `scripts/kaggle_pull_outputs.sh` — downloads trained weights after the job completes.

**Verification**:
```bash
cat kaggle/kernel-metadata.json | python3 -m json.tool
```

**Expected output**: Valid, pretty-printed JSON with no syntax errors.

---

### Milestone 6: Push Training Job to Kaggle GPU and Monitor

**Status**: 🟡 Running

**What to do**:

1. Push the classification training kernel:
   ```bash
   bash scripts/kaggle_push.sh classification
   ```

2. Monitor the run status:
   ```bash
   kaggle kernels status YOUR_KAGGLE_USERNAME/aerial-classification-training
   ```
   Statuses: `queued` → `running` → `complete` (or `error`)

3. Once complete, download outputs:
   ```bash
   bash scripts/kaggle_pull_outputs.sh classification
   ```

4. Verify the downloaded model weights:
   ```bash
   ls -la models/classification/custom_cnn/
   ```

**Verification**:
```bash
python3 -c "
import torch
state = torch.load('models/classification/custom_cnn/best_model.pth', map_location='cpu')
print(f'Keys: {len(state)} layers loaded')
print('✓ Model weights downloaded from Kaggle GPU successfully')
"
```

**Expected output**:
```
Keys: 24 layers loaded
✓ Model weights downloaded from Kaggle GPU successfully
```

**If it fails**:
- `"queued"` for too long → Kaggle GPU queue can take 5-15 minutes; check https://www.kaggle.com/code for your kernel status
- `"error"` status → Run `kaggle kernels output YOUR_KAGGLE_USERNAME/aerial-classification-training -p /tmp/kaggle-debug` and check the log files for Python errors
- Missing outputs → The kernel may still be running; check `kaggle kernels status` again

---

## Progress

| Date | Milestone | Status | Notes |
|------|-----------|--------|-------|
| 2026-04-13 | Plan created | ✅ Complete | ExecPlan authored with 6 milestones |
| | Milestone 1: Install & Auth | ✅ Complete | Kaggle CLI 2.0.1, token set via KAGGLE_API_TOKEN |
| | Milestone 2: Upload Dataset | ✅ Complete | ~228 MB uploaded, dataset live at aghasonemmanuel/aerial-bird-drone-detection |
| | Milestone 3: Classification Script | ✅ Complete | `kaggle/train_classification_kaggle.py` created |
| | Milestone 4: Detection Script | ✅ Complete | `kaggle/train_detection_kaggle.py` created |
| | Milestone 5: Metadata + Push/Pull | ✅ Complete | metadata + shell scripts created |
| | Milestone 6: Push & Monitor | 🟡 Running | v1 pushed, status RUNNING |

---

## Decision Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-04-13 | Use Kaggle **script** mode (not notebook) | Scripts are version-controllable plain `.py` files; notebooks add JSON cell noise to git diffs |
| 2026-04-13 | Upload `src/` inside the dataset | Lets Kaggle scripts import from `src.models.*` directly, keeping code DRY with local development |
| 2026-04-13 | Use `NvidiaTeslaT4` (not P100) | T4 is newer (Turing arch), has Tensor Cores for mixed precision, and is reliably available on free tier |
| 2026-04-13 | Single Kaggle dataset for both classification + detection | Avoids maintaining two datasets; both pipelines need overlapping source images |
| 2026-04-13 | Enable internet in kernels | Needed for `torchvision` to download pretrained weights (ResNet50, MobileNetV2, EfficientNet-B0) and `ultralytics` to download `yolov8m.pt` |
| 2026-04-13 | Self-contained training scripts in `kaggle/` | Even though `src/` is uploaded as dataset, having self-contained scripts makes debugging easier — no import path issues |

---

## Surprises & Discoveries

*(To be updated as milestones are implemented)*

---

## Outcomes & Retrospective

*(To be filled when all milestones are complete)*
