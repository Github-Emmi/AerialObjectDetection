# ExecPlan: Phase 1 — Data Validation & Integrity

**Created**: 2026-04-11
**Last Updated**: 2026-04-11
**Status**: 🟡 In Progress

---

## Why This Matters

Before training any model, we must confirm the dataset is intact: every image has a label, every label is valid, no files are corrupt, and no data leaks across splits. Skipping this step risks silent training failures or inflated metrics from leaked data. After this phase, running `python -m scripts.validate_dataset` prints a clean report confirming 3,400 detection images and 3,319 classification images with zero integrity errors.

---

## Prerequisites

- **Python**: 3.10+
- **OS**: macOS / Linux / Windows with WSL
- **GPU**: Not required for this phase
- **Working directory**: `object_detection_Dataset/` (the repository root)
- **Prior phases completed**: None — this is Phase 1

### Environment Setup

```bash
cd /path/to/object_detection_Dataset
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

---

## Repository Orientation

| Path | What It Does | Touch? |
|------|-------------|--------|
| `data.yaml` | YOLOv8 dataset config — class names `['Bird', 'drone']`, split paths | Read |
| `train/images/`, `train/labels/` | Detection training split (2,728 images) | Read |
| `valid/images/`, `valid/labels/` | Detection validation split (448 images) | Read |
| `test/images/`, `test/labels/` | Detection test split (224 images) | Read |
| `classification_dataset/train/{bird,drone}/` | Classification training split (2,662 images) | Read |
| `classification_dataset/valid/{bird,drone}/` | Classification validation split (442 images) | Read |
| `classification_dataset/test/{bird,drone}/` | Classification test split (215 images) | Read |
| `src/config.py` | Central configuration — paths, constants | CREATE |
| `src/data_validation.py` | All validation logic | CREATE |
| `scripts/validate_dataset.py` | CLI entry point — runs validation and prints report | CREATE |

### Key Definitions

- **Detection dataset**: Images at `{train,valid,test}/images/*.jpg` with YOLOv8 `.txt` labels at `{train,valid,test}/labels/*.txt`. Each label line: `<class_id> <x_center> <y_center> <width> <height>` with normalized coordinates in [0, 1]. Classes: `0 = Bird`, `1 = drone`.
- **Classification dataset**: Same source images reorganized into `classification_dataset/{split}/{class}/` folders. Labels are implicit from directory name. 81 background images (empty detection labels) are excluded.
- **Orphan**: An image without a matching label, or a label without a matching image (matched by filename stem).
- **Empty label**: A `.txt` file with 0 bytes or only whitespace — these are valid hard negatives for detection.
- **Data leakage**: When the same image appears in more than one split (train/valid/test), causing inflated validation/test metrics.

---

## Milestones

### Milestone 1: Project Scaffolding & Dependencies

**Status**: ⬜ Not Started

**What to do**:
1. Create `src/__init__.py`, `src/models/__init__.py` (empty files to make Python packages)
2. Create `requirements.txt` with all project dependencies
3. Create `.gitignore` for model weights, venv, caches
4. Create `src/config.py` with project paths and constants
5. Install dependencies in the virtual environment

**Verification**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); from PIL import Image; print('PIL OK'); import yaml; print('PyYAML OK')"
```

**Expected output**:
```
PyTorch 2.x.x
PIL OK
PyYAML OK
```

**If it fails**: Check that `.venv` is activated and `pip install -r requirements.txt` completed without errors.

---

### Milestone 2: Detection Dataset Validation

**Status**: ⬜ Not Started

**What to do**:
1. Create `src/data_validation.py` with functions:
   - `validate_detection_dataset()` — checks image-label pairing, label format, corrupt images, class distribution, empty labels
   - `check_duplicates_across_splits()` — hashes images to detect leakage
   - `validate_classification_dataset()` — checks image counts and readability
   - `cross_validate_datasets()` — confirms classification images are a strict subset of detection images
2. Each function returns a structured report dict

**Verification**:
```bash
python -c "from src.data_validation import validate_detection_dataset; print('Import OK')"
```

**Expected output**:
```
Import OK
```

---

### Milestone 3: CLI Runner & Full Validation

**Status**: ⬜ Not Started

**What to do**:
1. Create `scripts/validate_dataset.py` that imports from `src.data_validation`, runs all checks, and prints a formatted report
2. Run validation against the actual dataset
3. Confirm all counts match ground truth

**Verification**:
```bash
python -m scripts.validate_dataset
```

**Expected output** (approximate — exact format defined in implementation):
```
=== Detection Dataset Validation ===
Split: train  | Images: 2728 | Labels: 2728 | Orphans: 0 | Corrupt: 0 | Empty: 66
Split: valid  | Images: 448  | Labels: 448  | Orphans: 0 | Corrupt: 0 | Empty: 6
Split: test   | Images: 224  | Labels: 224  | Orphans: 0 | Corrupt: 0 | Empty: 9
Total: 3400 images, 81 empty labels
Bird bboxes: 1406 | Drone bboxes: 135 | Ratio: 10.4:1

=== Label Format Validation ===
Invalid labels: 0

=== Classification Dataset Validation ===
Split: train  | bird: 1414 | drone: 1248 | Total: 2662
Split: valid  | bird: 217  | drone: 225  | Total: 442
Split: test   | bird: 121  | drone: 94   | Total: 215
Total: 3319 images (= 3400 - 81 background)

=== Cross-Split Duplicate Check ===
Duplicates across splits: 0

=== Cross-Dataset Consistency ===
Classification is strict subset of detection: ✓

✅ Phase 1 PASSED — Dataset integrity confirmed
```

**If it fails**: The report will show exactly which files are problematic. Fix or investigate those specific files before proceeding to Phase 2.

---

## Progress

| Date | Milestone | Status | Notes |
|------|-----------|--------|-------|
| 2026-04-11 | Milestone 1 | ⬜ Not Started | — |
| 2026-04-11 | Milestone 2 | ⬜ Not Started | — |
| 2026-04-11 | Milestone 3 | ⬜ Not Started | — |

---

## Decision Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-04-11 | Use MD5 hashing for duplicate detection | Fast enough for 3,400 images; collision risk negligible for this dataset size |
| 2026-04-11 | Keep 81 empty-label images as hard negatives | They improve detector precision on background scenes; YOLOv8 handles them correctly |
| 2026-04-11 | Validate classification as subset of detection | Both datasets share byte-identical source images — this cross-check catches any inconsistency |

---

## Surprises & Discoveries

> Record unexpected findings here as they occur.

---

## Outcomes & Retrospective

> Fill in when all milestones are complete.

### What Was Achieved
- ...

### What Remains
- ...

### Lessons Learned
- ...

### Observable Proof

```bash
python -m scripts.validate_dataset
```

```
✅ Phase 1 PASSED — Dataset integrity confirmed
```
