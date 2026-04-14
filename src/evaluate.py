"""Model evaluation and comparison report generation.

Evaluates trained classification models (Custom CNN, ResNet50, MobileNetV2,
EfficientNet-B0) and the YOLOv8m detector on their respective test sets.
Produces per-model metrics, confusion matrices, ROC curves, and a final
comparison CSV.

Usage:
    python -m src.evaluate                          # All classifiers + detection
    python -m src.evaluate --models custom_cnn      # Single classifier
    python -m src.evaluate --detection-only         # YOLOv8m only
    python -m src.evaluate --skip-detection         # Classifiers only
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    PROJECT_ROOT,
    ClassificationConfig,
    DetectionConfig,
)
from src.models.custom_cnn import AerialCNN
from src.models.transfer_learning import create_transfer_model
from src.preprocessing import get_classification_loaders
from src.utils import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
    plot_training_curves,
)

# ── Constants ────────────────────────────────────────────────
CLASSIFIER_NAMES = ["custom_cnn", "resnet50", "mobilenet_v2", "efficientnet_b0"]
CLASS_LIST = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]  # ["Bird", "drone"]
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFUSION_DIR = REPORTS_DIR / "confusion_matrices"
CURVES_DIR = REPORTS_DIR / "training_curves"
ROC_DIR = REPORTS_DIR / "roc_curves"


# ── Model Building ───────────────────────────────────────────

def _build_model(name: str) -> torch.nn.Module:
    if name == "custom_cnn":
        return AerialCNN(num_classes=NUM_CLASSES)
    return create_transfer_model(backbone=name, num_classes=NUM_CLASSES, freeze_ratio=0.0)


def _load_classifier(name: str, weights_dir: Path) -> torch.nn.Module:
    """Load a trained classifier from saved weights."""
    weights_path = weights_dir / name / "best_model.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = _build_model(name)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Classification Evaluation ────────────────────────────────

@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Run inference on test set and compute all classification metrics.

    Returns dict with keys: accuracy, precision, recall, f1, roc_auc,
    y_true, y_pred, y_probs, inference_ms
    """
    model.to(device).eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    start = time.perf_counter()
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # P(drone)
    elapsed = time.perf_counter() - start

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    n_samples = len(y_true)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "inference_ms": (elapsed / n_samples * 1000) if n_samples else 0.0,
        "n_samples": n_samples,
    }


def run_classification_eval(
    model_names: list[str],
    weights_dir: Path,
    cfg: ClassificationConfig,
) -> list[dict]:
    """Evaluate one or more classification models and save reports."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders = get_classification_loaders(cfg, num_workers=0)
    if "test" not in loaders:
        raise RuntimeError("Test split not found in classification dataset")

    test_loader = loaders["test"]
    n_test = len(test_loader.dataset)
    print(f"Test images: {n_test}")

    results: list[dict] = []

    for name in model_names:
        print(f"\n{'─'*60}")
        print(f"Evaluating: {name}")
        print(f"{'─'*60}")

        try:
            model = _load_classifier(name, weights_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        total_params = sum(p.numel() for p in model.parameters())
        metrics = evaluate_classifier(model, test_loader, device)

        # Print results
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} (weighted)")
        print(f"  Recall:    {metrics['recall']:.4f} (weighted)")
        print(f"  F1-Score:  {metrics['f1']:.4f} (weighted)")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Inference: {metrics['inference_ms']:.1f} ms/image (CPU)")

        # Classification report
        report = classification_report(
            metrics["y_true"], metrics["y_pred"],
            target_names=CLASS_LIST, zero_division=0,
        )
        print(f"\n{report}")

        # Save confusion matrix plot
        cm_path = CONFUSION_DIR / f"{name}.png"
        plot_confusion_matrix(
            metrics["y_true"], metrics["y_pred"], CLASS_LIST,
            save_path=cm_path, title=f"{name} — Confusion Matrix",
        )
        print(f"  Saved: {cm_path.relative_to(PROJECT_ROOT)}")

        # Save ROC curve
        roc_path = ROC_DIR / f"{name}.png"
        plot_roc_curve(
            metrics["y_true"], metrics["y_probs"],
            save_path=roc_path, title=f"{name} — ROC Curve",
        )
        print(f"  Saved: {roc_path.relative_to(PROJECT_ROOT)}")

        # Save training curves (if log exists)
        log_csv = weights_dir / name / "training_log.csv"
        if log_csv.exists():
            curves_path = CURVES_DIR / f"{name}.png"
            plot_training_curves(
                log_csv, save_path=curves_path,
                title=f"{name} — Training Curves",
            )
            print(f"  Saved: {curves_path.relative_to(PROJECT_ROOT)}")

        # Save text report
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"{name}_results.txt"
        with open(report_path, "w") as f:
            f.write(f"model={name}\n")
            f.write(f"accuracy={metrics['accuracy']:.4f}\n")
            f.write(f"precision={metrics['precision']:.4f}\n")
            f.write(f"recall={metrics['recall']:.4f}\n")
            f.write(f"f1={metrics['f1']:.4f}\n")
            f.write(f"roc_auc={metrics['roc_auc']:.4f}\n")
            f.write(f"inference_ms={metrics['inference_ms']:.1f}\n")
            f.write(f"params={total_params}\n")
            f.write(f"test_samples={metrics['n_samples']}\n")
            f.write(f"\n{report}\n")
        print(f"  Saved: {report_path.relative_to(PROJECT_ROOT)}")

        results.append({
            "model": name,
            "type": "classification",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "params": total_params,
            "inference_ms": metrics["inference_ms"],
        })

    return results


# ── Detection Evaluation ─────────────────────────────────────

def run_detection_eval(det_cfg: DetectionConfig) -> dict | None:
    """Evaluate YOLOv8m on test set and report per-class AP."""
    weights_path = det_cfg.model_save_dir / "best.pt"
    if not weights_path.exists():
        print(f"Detection weights not found: {weights_path}")
        return None

    from ultralytics import YOLO

    print(f"\n{'─'*60}")
    print("Evaluating: YOLOv8m (detection)")
    print(f"{'─'*60}")
    print(f"Weights: {weights_path}")

    model = YOLO(str(weights_path))

    # Run validation on test split
    metrics = model.val(
        data=str(det_cfg.data_yaml),
        split="test",
        imgsz=det_cfg.imgsz,
        batch=det_cfg.batch_size,
        device="cpu",
        verbose=False,
    )

    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    # Per-class AP
    bird_ap50 = metrics.box.ap50[0] if len(metrics.box.ap50) > 0 else 0.0
    drone_ap50 = metrics.box.ap50[1] if len(metrics.box.ap50) > 1 else 0.0

    # Per-class precision/recall (from metrics.box.p and metrics.box.r)
    bird_p = metrics.box.p[0] if len(metrics.box.p) > 0 else 0.0
    drone_p = metrics.box.p[1] if len(metrics.box.p) > 1 else 0.0
    bird_r = metrics.box.r[0] if len(metrics.box.r) > 0 else 0.0
    drone_r = metrics.box.r[1] if len(metrics.box.r) > 1 else 0.0

    overall_p = float(np.mean(metrics.box.p)) if len(metrics.box.p) else 0.0
    overall_r = float(np.mean(metrics.box.r)) if len(metrics.box.r) else 0.0

    print(f"  mAP@0.5:      {map50:.4f}")
    print(f"  mAP@0.5:0.95: {map50_95:.4f}")
    print(f"  Bird  AP@0.5:  {bird_ap50:.4f}  P: {bird_p:.4f}  R: {bird_r:.4f}")
    print(f"  Drone AP@0.5:  {drone_ap50:.4f}  P: {drone_p:.4f}  R: {drone_r:.4f}")

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "yolov8m_results.txt"
    with open(report_path, "w") as f:
        f.write("model=yolov8m\n")
        f.write(f"mAP50={map50:.4f}\n")
        f.write(f"mAP50_95={map50_95:.4f}\n")
        f.write(f"bird_ap50={bird_ap50:.4f}\n")
        f.write(f"drone_ap50={drone_ap50:.4f}\n")
        f.write(f"bird_precision={bird_p:.4f}\n")
        f.write(f"bird_recall={bird_r:.4f}\n")
        f.write(f"drone_precision={drone_p:.4f}\n")
        f.write(f"drone_recall={drone_r:.4f}\n")
        f.write(f"overall_precision={overall_p:.4f}\n")
        f.write(f"overall_recall={overall_r:.4f}\n")
    print(f"  Saved: {report_path.relative_to(PROJECT_ROOT)}")

    return {
        "model": "yolov8m",
        "type": "detection",
        "accuracy": map50,  # Use mAP50 in the comparison column
        "precision": overall_p,
        "recall": overall_r,
        "f1": 0.0,  # Not directly applicable for detection
        "roc_auc": 0.0,
        "params": 25_900_000,  # YOLOv8m standard param count
        "inference_ms": 0.0,  # Would need timed inference loop
        "mAP50": map50,
        "mAP50_95": map50_95,
        "bird_ap50": bird_ap50,
        "drone_ap50": drone_ap50,
    }


# ── Comparison Report ────────────────────────────────────────

def generate_comparison(results: list[dict]) -> None:
    """Write the model comparison CSV and print summary table."""
    if not results:
        print("No results to compare.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "model_comparison.csv"

    fieldnames = [
        "model", "type", "accuracy", "precision", "recall",
        "f1", "roc_auc", "params", "inference_ms",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Print table
    print(f"\n{'='*80}")
    print("MODEL COMPARISON REPORT")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Acc/mAP50':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'Params':>12} {'ms/img':>8}")
    print(f"{'─'*80}")

    for r in results:
        params_str = _format_params(r["params"])
        f1_str = f"{r['f1']:.4f}" if r["f1"] > 0 else "—"
        print(f"{r['model']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {f1_str:>10} {params_str:>12} "
              f"{r['inference_ms']:>8.1f}")

    print(f"{'─'*80}")
    print(f"Saved: {csv_path.relative_to(PROJECT_ROOT)}")

    # Best classifier by F1
    classifiers = [r for r in results if r["type"] == "classification" and r["f1"] > 0]
    if classifiers:
        best = max(classifiers, key=lambda r: r["f1"])
        print(f"\n★ Best classifier (by F1): {best['model']} — F1={best['f1']:.4f}")

    # Detection summary
    detectors = [r for r in results if r["type"] == "detection"]
    if detectors:
        d = detectors[0]
        print(f"★ Detection model: {d['model']} — mAP@0.5={d.get('mAP50', d['accuracy']):.4f}")

    # Plot comparison bar chart
    clf_results = [r for r in results if r["type"] == "classification"]
    if clf_results:
        plot_model_comparison(
            [r["model"] for r in clf_results],
            [r["f1"] for r in clf_results],
            metric_name="F1-Score (weighted)",
            save_path=REPORTS_DIR / "f1_comparison.png",
            title="Classification Model Comparison — F1-Score",
        )
        print(f"Saved: reports/f1_comparison.png")


def _format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--models", nargs="+", default=CLASSIFIER_NAMES,
        choices=CLASSIFIER_NAMES,
        help="Classification models to evaluate (default: all 4)",
    )
    parser.add_argument(
        "--detection-only", action="store_true",
        help="Only evaluate YOLOv8m detection model",
    )
    parser.add_argument(
        "--skip-detection", action="store_true",
        help="Skip detection evaluation",
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=None,
        help="Override classification weights directory",
    )
    args = parser.parse_args()

    cls_cfg = ClassificationConfig()
    det_cfg = DetectionConfig()
    weights_dir = args.weights_dir or cls_cfg.model_save_dir

    all_results: list[dict] = []

    if not args.detection_only:
        print("=" * 60)
        print("CLASSIFICATION EVALUATION")
        print("=" * 60)
        cls_results = run_classification_eval(args.models, weights_dir, cls_cfg)
        all_results.extend(cls_results)

    if not args.skip_detection:
        det_result = run_detection_eval(det_cfg)
        if det_result:
            all_results.append(det_result)

    generate_comparison(all_results)


if __name__ == "__main__":
    main()
