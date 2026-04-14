# ExecPlan: Phase 7 — Streamlit Deployment

**Created**: 2026-04-13
**Last Updated**: 2026-04-13
**Status**: ✅ Complete

---

## Why This Matters

After this work, a user can run `streamlit run app/app.py` and interact with a web UI that lets them upload aerial images, classify them as Bird or Drone using any of the 4 trained classifiers, and detect + localize objects with YOLOv8m bounding box overlays. This is the public-facing deliverable of the entire ML pipeline.

---

## Prerequisites

- **Python**: 3.12 (pyenv-managed venv at `venv/`)
- **OS**: macOS (Intel, no local GPU — all inference on CPU)
- **Working directory**: `/Volumes/EmmiDev256G/Projects/object_detection_Dataset`
- **Prior phases completed**:
  - Phase 6 ✅: All 5 models evaluated, comparison report generated
- **Model weights available**:
  - `models/classification/{custom_cnn,resnet50,mobilenet_v2,efficientnet_b0}/best_model.pth`
  - `models/detection/best.pt` (YOLOv8m, 50MB)
- **Dependencies**: streamlit>=1.30.0, torch, torchvision, ultralytics, Pillow (all in requirements.txt)

### What Already Exists

| Path | Status |
|------|--------|
| `app/` | Directory exists, empty |
| `app/components/` | Directory exists, empty |
| `app/assets/` | Directory exists, empty |
| `src/config.py` | ✅ ClassificationConfig, DetectionConfig, CLASS_NAMES |
| `src/models/custom_cnn.py` | ✅ AerialCNN class |
| `src/models/transfer_learning.py` | ✅ create_transfer_model() |

### Architecture Decision

- **Best classifier for default**: MobileNetV2 (F1=0.9860, 2.2M params, 63ms/img — smallest + fastest at top F1)
- **All 4 classifiers available** via dropdown selector
- **YOLOv8m** for detection (loaded via ultralytics YOLO class)
- Sidebar navigation: Classification mode vs Detection mode
- No GPU required — CPU inference is fast enough for single-image upload

---

## Repository Orientation

```
app/
├── app.py                   # CREATE: Main Streamlit entry point, page config, sidebar navigation
├── components/
│   ├── __init__.py          # CREATE: Package init
│   ├── classifier_ui.py     # CREATE: Classification mode — upload, model select, prediction, confidence chart
│   └── detector_ui.py       # CREATE: Detection mode — upload, YOLOv8m inference, bbox overlay display
└── assets/                  # For sample images (optional)
```

---

## Milestones

### Milestone 1: Build app/app.py — Main Entry Point
- Page config (title, icon, layout)
- Sidebar: mode selector (Classification / Detection), project info
- Route to classifier_ui or detector_ui based on selection

### Milestone 2: Build classifier_ui.py
- Image upload (jpg/png)
- Model selector dropdown (4 classifiers, default: MobileNetV2)
- Preprocessing (resize 224×224, ImageNet normalize)
- Inference + softmax confidence
- Display: uploaded image, prediction label, confidence bar chart

### Milestone 3: Build detector_ui.py
- Image upload (jpg/png)
- Load YOLOv8m from models/detection/best.pt
- Run inference, draw bounding boxes on image
- Display: annotated image, detection table (class, confidence, bbox coords)
- Confidence threshold slider

### Milestone 4: Test End-to-End
- Run `streamlit run app/app.py`
- Test both modes with sample images from test set
- Verify model loading, inference, display

---

## Progress

| Date | Milestone | Status | Notes |
|------|-----------|--------|-------|
| 2026-04-13 | Plan created | ✅ Complete | |
| 2026-04-13 | M1: app.py | ✅ Complete | Sidebar nav, mode routing |
| 2026-04-13 | M2: classifier_ui.py | ✅ Complete | 4-model dropdown, confidence chart |
| 2026-04-13 | M3: detector_ui.py | ✅ Complete | YOLOv8m, bbox overlay, detection table |
| 2026-04-13 | M4: Test E2E | ✅ Complete | All 5 models load OK, app runs on :8501 |

---

## Decision Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-04-13 | Default classifier: MobileNetV2 | Tied best F1 (0.9860), smallest (2.2M), fastest (63ms) |
| 2026-04-13 | CPU-only inference | Single image per request, latency acceptable on CPU |
| 2026-04-13 | Two-mode UI (Classification/Detection) | Matches project deliverables in project_summary.md |

---

## Surprises & Discoveries

- ultralytics `result.plot()` returns BGR numpy array — needed `[..., ::-1]` for PIL RGB conversion
- `st.cache_resource` is ideal for model caching across reruns — no reload overhead
- `weights_only=True` in `torch.load()` is required for security (PyTorch 2.x default)

---

## Outcomes & Retrospective

**Results**: Fully functional two-mode Streamlit UI at `http://localhost:8501`
- Classification: 4 models selectable, default MobileNetV2, confidence bar chart
- Detection: YOLOv8m with adjustable confidence threshold, bbox overlay, detection table
- All 5 models verified loading correctly; smoke test passed (Bird classified at 100%, drones detected at 84-89%)

**What went well**: Clean separation into components, cached model loading, straightforward image processing pipeline

**Lessons**: Keep UI components stateless — let Streamlit's rerun model + `@st.cache_resource` handle state
