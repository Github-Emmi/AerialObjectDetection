# Project Context — Aerial Object Classification & Detection

> This file is the single source of verified facts about the project. Every number here was confirmed by running commands against the actual filesystem on 2026-04-11. If documentation (project_summary.md, README.roboflow.txt) contradicts this file, **this file is correct**.

## Repository Root

```
/Volumes/EmmiDev256G/Projects/object_detection_Dataset/
```

## What This Project Does

This project builds a deep-learning system that looks at aerial images and answers two questions:

1. **Classification**: "Is this image a Bird or a Drone?" — answered by a Custom CNN or a transfer-learning model (ResNet50, MobileNetV2, EfficientNet-B0), outputting a single label + confidence score.
2. **Object Detection**: "Where in this image are the Birds and/or Drones?" — answered by YOLOv8m, outputting bounding boxes with class labels and confidence scores.

The final product is a Streamlit web app (containerized in Docker) where a user uploads an image and gets either a classification label or annotated detection bounding boxes.

## Datasets — The Single Most Important Fact

**The classification and detection datasets are the same images.** Byte-for-byte identical `.jpg` files exist in both locations. The classification dataset is simply the detection dataset reorganized into `bird/` and `drone/` folders, with 81 background images (empty label files) excluded.

### Detection Dataset (Primary)

Located at the repository root in `train/`, `valid/`, `test/` directories.

| Split | Images | Labels | Empty Labels | Bird Bboxes | Drone Bboxes |
|-------|--------|--------|-------------|-------------|-------------|
| Train | 2,728 | 2,728 | 66 | 1,153 | 109 |
| Valid | 448 | 448 | 6 | 198 | 24 |
| Test | 224 | 224 | 9 | 57 | 2 |
| **Total** | **3,400** | **3,400** | **81** | **1,406** | **135** |

- Format: YOLOv8 `.txt` files — each line is `<class_id> <x_center> <y_center> <width> <height>` with normalized coordinates in [0, 1]
- Classes: `0 = Bird`, `1 = drone` (defined in `data.yaml`)
- Bounding box ratio: **10.4:1 Bird:Drone** (severe imbalance at bbox level)
- Image-level ratio: ~1.13:1 Bird:Drone (roughly balanced)
- Zero images contain both classes simultaneously
- 81 images have empty `.txt` files (background/hard negatives — no objects present)
- Images are 416×416 pixels (Roboflow stretched from originals)
- Augmentation: 50% horizontal flip, producing ~3 versions per source image

### Classification Dataset (Derived)

Located at `classification_dataset/` with `train/`, `valid/`, `test/` subdirectories, each containing `bird/` and `drone/` folders.

| Split | Bird | Drone | Total |
|-------|------|-------|-------|
| Train | 1,414 | 1,248 | 2,662 |
| Valid | 217 | 225 | 442 |
| Test | 121 | 94 | 215 |
| **Total** | **1,752** | **1,567** | **3,319** |

- Format: `.jpg` files sorted into class folders (PyTorch `ImageFolder` compatible)
- Labels: Implicit from directory name — no annotation files
- Total: 3,319 = 3,400 detection images − 81 background images

### How the Numbers Relate

```
Detection images with Bird labels  = Classification bird images  (exact match per split)
Detection images with Drone labels = Classification drone images (exact match per split)
Detection empty-label images       = Not in classification dataset at all
3,400 − 81 = 3,319 ✓
```

## Technology Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| DL Framework | PyTorch only | User requirement — no TensorFlow/Keras |
| Detection model | YOLOv8m (medium) | User selected; 25.9M params, good accuracy for 2-class task |
| Classification models | Custom CNN + ResNet50 + MobileNetV2 + EfficientNet-B0 | Compare custom vs transfer learning |
| Deployment | Streamlit + Docker | User selected Docker over local-only or cloud |
| Class names | `['Bird', 'drone']` as-is | User explicitly chose not to normalize capitalization |
| Background images | Keep as hard negatives | User delegated to agent; these improve detector precision |
| Roboflow re-annotation | NOT needed | Both datasets already share annotations |

## Key File Paths

| File | Purpose |
|------|---------|
| `data.yaml` | YOLOv8 dataset config (paths, class names, Roboflow metadata) |
| `project_summary.md` | Business requirements and project specification |
| `.github/agents/aerial-detection.agent.md` | Full technical architecture (8 phases) |
| `.github/skills/exec-plan/SKILL.md` | This skill — ExecPlan methodology |
| `train/images/`, `train/labels/` | Detection training data |
| `valid/images/`, `valid/labels/` | Detection validation data |
| `test/images/`, `test/labels/` | Detection test data |
| `classification_dataset/train/{bird,drone}/` | Classification training data |
| `classification_dataset/valid/{bird,drone}/` | Classification validation data |
| `classification_dataset/test/{bird,drone}/` | Classification test data |
