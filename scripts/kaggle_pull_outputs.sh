#!/usr/bin/env bash
# ── kaggle_pull_outputs.sh ──────────────────────────────────
# Download trained model weights from a completed Kaggle kernel.
#
# Usage:
#   ./scripts/kaggle_pull_outputs.sh              # Pull classification outputs
#   ./scripts/kaggle_pull_outputs.sh detection     # Pull detection outputs
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KAGGLE_DIR="$PROJECT_ROOT/kaggle"

MODE="${1:-classification}"

# Get kernel slug from metadata
KERNEL_SLUG=$(python3 -c "
import json
with open('$KAGGLE_DIR/kernel-metadata.json') as f:
    meta = json.load(f)
slug = meta['id']
if '$MODE' == 'detection':
    slug = slug.replace('aerial-classification-training', 'aerial-detection-training')
print(slug)
")

# Output directory
OUTPUT_DIR="$PROJECT_ROOT/models/${MODE}_kaggle"
mkdir -p "$OUTPUT_DIR"

echo "==> Checking kernel status: $KERNEL_SLUG"
kaggle kernels status "$KERNEL_SLUG"

echo "==> Downloading outputs to: $OUTPUT_DIR"
kaggle kernels output "$KERNEL_SLUG" -p "$OUTPUT_DIR"

echo "==> Contents:"
ls -lhR "$OUTPUT_DIR"

echo ""
echo "Done! Trained weights saved to $OUTPUT_DIR"
