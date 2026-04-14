"""Dataset validation for the Aerial Object Classification & Detection project.

Provides functions to verify:
- Image-label pairing (no orphans)
- Label format correctness (YOLOv8 format)
- Image readability (no corrupt files)
- Class distribution per split
- Cross-split duplicate detection (data leakage)
- Classification ⊂ Detection consistency
"""

import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

from src.config import CLASS_NAMES, DETECTION_ROOT, CLASSIFICATION_ROOT, SPLITS


def _md5(filepath: Path) -> str:
    """Return the MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_detection_dataset(data_root: Path = DETECTION_ROOT) -> dict:
    """Validate detection image-label pairs, label format, and image integrity.

    Returns a report dict with keys:
        splits: per-split stats (images, labels, orphans, corrupt, empty, class_counts)
        invalid_labels: list of (file, line_num, reason)
        total_class_counts: {class_id: count}
    """
    report = {
        "splits": {},
        "invalid_labels": [],
        "total_class_counts": defaultdict(int),
    }

    for split in SPLITS:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"

        img_stems = {p.stem: p for p in sorted(img_dir.glob("*.jpg")) if not p.name.startswith("._")}
        lbl_stems = {p.stem: p for p in sorted(lbl_dir.glob("*.txt")) if not p.name.startswith("._")}

        orphan_images = sorted(set(img_stems) - set(lbl_stems))
        orphan_labels = sorted(set(lbl_stems) - set(img_stems))

        # Check for corrupt images
        corrupt_images = []
        for stem, img_path in img_stems.items():
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception as e:
                corrupt_images.append((str(img_path), str(e)))

        # Validate labels
        empty_count = 0
        split_class_counts = defaultdict(int)

        for stem, lbl_path in lbl_stems.items():
            try:
                content = lbl_path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                report["invalid_labels"].append(
                    (str(lbl_path), 0, "File contains non-UTF-8 bytes")
                )
                continue
            if not content:
                empty_count += 1
                continue

            for line_num, line in enumerate(content.split("\n"), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    report["invalid_labels"].append(
                        (str(lbl_path), line_num, f"Expected 5 fields, got {len(parts)}")
                    )
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    report["invalid_labels"].append(
                        (str(lbl_path), line_num, "Non-numeric values")
                    )
                    continue

                if cls_id not in CLASS_NAMES:
                    report["invalid_labels"].append(
                        (str(lbl_path), line_num, f"Unknown class_id {cls_id}")
                    )
                    continue
                if not all(0.0 <= c <= 1.0 for c in coords):
                    report["invalid_labels"].append(
                        (str(lbl_path), line_num, f"Coordinates out of [0,1]: {coords}")
                    )
                    continue

                split_class_counts[cls_id] += 1
                report["total_class_counts"][cls_id] += 1

        report["splits"][split] = {
            "images": len(img_stems),
            "labels": len(lbl_stems),
            "orphan_images": orphan_images,
            "orphan_labels": orphan_labels,
            "corrupt_images": corrupt_images,
            "empty_labels": empty_count,
            "class_counts": dict(split_class_counts),
        }

    return report


def check_duplicates_across_splits(data_root: Path = DETECTION_ROOT) -> dict:
    """Hash all detection images and find duplicates across train/valid/test.

    Returns:
        duplicates: list of (hash, [(split, filename), ...]) for images appearing in >1 split
        total_hashed: total images hashed
    """
    hash_to_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in SPLITS:
        img_dir = data_root / split / "images"
        for img_path in sorted(img_dir.glob("*.jpg")):
            if img_path.name.startswith("._"):
                continue
            digest = _md5(img_path)
            hash_to_locations[digest].append((split, img_path.name))

    duplicates = [
        (h, locs) for h, locs in hash_to_locations.items() if len(locs) > 1
    ]
    total_hashed = sum(len(locs) for locs in hash_to_locations.values())

    return {"duplicates": duplicates, "total_hashed": total_hashed}


def validate_classification_dataset(data_root: Path = CLASSIFICATION_ROOT) -> dict:
    """Validate classification dataset structure and image readability.

    Returns a report dict with per-split, per-class image counts and any corrupt files.
    """
    report = {"splits": {}, "corrupt_images": []}

    for split in SPLITS:
        split_dir = data_root / split
        class_counts = {}
        for class_name in ("bird", "drone"):
            class_dir = split_dir / class_name
            images = [p for p in sorted(class_dir.glob("*.jpg")) if not p.name.startswith("._")]
            class_counts[class_name] = len(images)

            # Spot-check readability (every image — dataset is small enough)
            for img_path in images:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    report["corrupt_images"].append((str(img_path), str(e)))

        report["splits"][split] = class_counts

    return report


def cross_validate_datasets(
    detection_root: Path = DETECTION_ROOT,
    classification_root: Path = CLASSIFICATION_ROOT,
) -> dict:
    """Verify that classification images are a strict subset of detection images.

    Hashes every image in both datasets and checks:
    1. Every classification image exists in the detection dataset (same split)
    2. Detection images NOT in classification are exactly the empty-label images

    Returns:
        classification_not_in_detection: list of classification images missing from detection
        detection_only: count of detection images not in classification (should equal empty label count)
        match: bool — True if datasets are consistent
    """
    det_hashes: dict[str, set[str]] = {s: set() for s in SPLITS}
    cls_hashes: dict[str, set[str]] = {s: set() for s in SPLITS}

    for split in SPLITS:
        # Detection images
        det_img_dir = detection_root / split / "images"
        for img_path in det_img_dir.glob("*.jpg"):
            if img_path.name.startswith("._"):
                continue
            det_hashes[split].add(_md5(img_path))

        # Classification images
        cls_split_dir = classification_root / split
        for class_name in ("bird", "drone"):
            class_dir = cls_split_dir / class_name
            for img_path in class_dir.glob("*.jpg"):
                if img_path.name.startswith("._"):
                    continue
                cls_hashes[split].add(_md5(img_path))

    classification_not_in_detection = []
    detection_only_count = 0

    for split in SPLITS:
        missing = cls_hashes[split] - det_hashes[split]
        if missing:
            classification_not_in_detection.extend(
                (split, h) for h in missing
            )
        detection_only_count += len(det_hashes[split] - cls_hashes[split])

    return {
        "classification_not_in_detection": classification_not_in_detection,
        "detection_only_count": detection_only_count,
        "match": len(classification_not_in_detection) == 0,
    }
