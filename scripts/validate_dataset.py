"""Standalone CLI runner for dataset validation.

Usage:
    python -m scripts.validate_dataset
    python -m scripts.validate_dataset --skip-cross-dataset
"""

import argparse
import sys
import time

from src.config import CLASS_NAMES, DETECTION_ROOT, CLASSIFICATION_ROOT
from src.data_validation import (
    validate_detection_dataset,
    check_duplicates_across_splits,
    validate_classification_dataset,
    cross_validate_datasets,
)


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_detection_report(report: dict) -> bool:
    """Print detection validation results. Returns True if all checks pass."""
    print_header("Detection Dataset Validation")
    passed = True

    total_images = 0
    total_empty = 0
    for split in ("train", "valid", "test"):
        s = report["splits"][split]
        total_images += s["images"]
        total_empty += s["empty_labels"]
        orphan_flag = ""
        if s["orphan_images"] or s["orphan_labels"]:
            orphan_flag = " ⚠️"
            passed = False
        corrupt_flag = " ⚠️" if s["corrupt_images"] else ""
        if s["corrupt_images"]:
            passed = False

        print(
            f"  Split: {split:<5} | Images: {s['images']:<5} | Labels: {s['labels']:<5} "
            f"| Orphans: {len(s['orphan_images']) + len(s['orphan_labels']):<3}{orphan_flag} "
            f"| Corrupt: {len(s['corrupt_images']):<3}{corrupt_flag} "
            f"| Empty: {s['empty_labels']}"
        )

    bird_total = report["total_class_counts"].get(0, 0)
    drone_total = report["total_class_counts"].get(1, 0)
    ratio = bird_total / drone_total if drone_total > 0 else float("inf")

    print(f"\n  Total: {total_images} images, {total_empty} empty labels")
    print(
        f"  {CLASS_NAMES[0]} bboxes: {bird_total} | "
        f"{CLASS_NAMES[1]} bboxes: {drone_total} | "
        f"Ratio: {ratio:.1f}:1"
    )

    if report["invalid_labels"]:
        passed = False
        print(f"\n  ⚠️  Invalid labels: {len(report['invalid_labels'])}")
        for path, line, reason in report["invalid_labels"][:10]:
            print(f"     {path}:{line} — {reason}")
        if len(report["invalid_labels"]) > 10:
            print(f"     ... and {len(report['invalid_labels']) - 10} more")
    else:
        print(f"\n  ✓ Label format: All labels valid")

    # Print orphan details if any
    for split in ("train", "valid", "test"):
        s = report["splits"][split]
        if s["orphan_images"]:
            print(f"\n  ⚠️  Orphan images in {split}: {s['orphan_images'][:5]}")
        if s["orphan_labels"]:
            print(f"\n  ⚠️  Orphan labels in {split}: {s['orphan_labels'][:5]}")
        if s["corrupt_images"]:
            print(f"\n  ⚠️  Corrupt images in {split}:")
            for path, err in s["corrupt_images"][:5]:
                print(f"     {path}: {err}")

    return passed


def print_classification_report(report: dict) -> bool:
    """Print classification validation results. Returns True if all checks pass."""
    print_header("Classification Dataset Validation")
    passed = True

    total = 0
    for split in ("train", "valid", "test"):
        counts = report["splits"][split]
        split_total = sum(counts.values())
        total += split_total
        print(
            f"  Split: {split:<5} | bird: {counts.get('bird', 0):<5} "
            f"| drone: {counts.get('drone', 0):<5} | Total: {split_total}"
        )

    print(f"\n  Total: {total} images (= 3400 - 81 background)")

    if report["corrupt_images"]:
        passed = False
        print(f"\n  ⚠️  Corrupt images: {len(report['corrupt_images'])}")
        for path, err in report["corrupt_images"][:5]:
            print(f"     {path}: {err}")
    else:
        print(f"  ✓ All images readable")

    return passed


def print_duplicate_report(report: dict) -> bool:
    """Print cross-split duplicate results. Returns True if no duplicates."""
    print_header("Cross-Split Duplicate Check")

    if report["duplicates"]:
        print(f"  ⚠️  Duplicates across splits: {len(report['duplicates'])}")
        for h, locs in report["duplicates"][:5]:
            locs_str = ", ".join(f"{s}/{f}" for s, f in locs)
            print(f"     {h[:12]}... → {locs_str}")
        return False
    else:
        print(f"  ✓ Duplicates across splits: 0 (hashed {report['total_hashed']} images)")
        return True


def print_cross_dataset_report(report: dict) -> bool:
    """Print cross-dataset consistency results. Returns True if consistent."""
    print_header("Cross-Dataset Consistency")

    if report["classification_not_in_detection"]:
        print(
            f"  ⚠️  Classification images NOT in detection: "
            f"{len(report['classification_not_in_detection'])}"
        )
        return False
    else:
        print(f"  ✓ Classification is strict subset of detection")
        print(f"  ✓ Detection-only images (background): {report['detection_only_count']}")
        return report["match"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate aerial detection & classification datasets")
    parser.add_argument(
        "--skip-cross-dataset",
        action="store_true",
        help="Skip the cross-dataset hash comparison (slow on large datasets)",
    )
    args = parser.parse_args()

    all_passed = True
    start = time.time()

    # 1. Detection validation
    print("\nRunning detection dataset validation...")
    det_report = validate_detection_dataset(DETECTION_ROOT)
    if not print_detection_report(det_report):
        all_passed = False

    # 2. Classification validation
    print("\nRunning classification dataset validation...")
    cls_report = validate_classification_dataset(CLASSIFICATION_ROOT)
    if not print_classification_report(cls_report):
        all_passed = False

    # 3. Cross-split duplicate check
    print("\nRunning cross-split duplicate check (hashing all images)...")
    dup_report = check_duplicates_across_splits(DETECTION_ROOT)
    if not print_duplicate_report(dup_report):
        all_passed = False

    # 4. Cross-dataset consistency (optional, uses hashing)
    if not args.skip_cross_dataset:
        print("\nRunning cross-dataset consistency check...")
        cross_report = cross_validate_datasets(DETECTION_ROOT, CLASSIFICATION_ROOT)
        if not print_cross_dataset_report(cross_report):
            all_passed = False
    else:
        print("\n  Skipping cross-dataset consistency check (--skip-cross-dataset)")

    elapsed = time.time() - start

    # Final verdict
    print(f"\n{'=' * 60}")
    if all_passed:
        print(f"  ✅ Phase 1 PASSED — Dataset integrity confirmed ({elapsed:.1f}s)")
    else:
        print(f"  ❌ Phase 1 FAILED — See issues above ({elapsed:.1f}s)")
    print(f"{'=' * 60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
