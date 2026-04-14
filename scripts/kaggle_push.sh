#!/usr/bin/env bash
# ── kaggle_push.sh ──────────────────────────────────────────
# Push a Kaggle kernel and monitor its status until completion.
#
# Usage:
#   ./scripts/kaggle_push.sh              # Push classification kernel
#   ./scripts/kaggle_push.sh detection    # Push detection kernel
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KAGGLE_DIR="$PROJECT_ROOT/kaggle"

MODE="${1:-classification}"

# Swap kernel metadata based on mode
if [[ "$MODE" == "detection" ]]; then
    echo "==> Preparing detection kernel push..."
    # Temporarily update kernel-metadata.json for detection
    cp "$KAGGLE_DIR/kernel-metadata.json" "$KAGGLE_DIR/kernel-metadata.json.bak"
    python3 -c "
import json
with open('$KAGGLE_DIR/kernel-metadata.json') as f:
    meta = json.load(f)
meta['id'] = meta['id'].replace('aerial-classification-training', 'aerial-detection-training')
meta['title'] = 'Aerial Detection Training'
meta['code_file'] = 'train_detection_kaggle.py'
with open('$KAGGLE_DIR/kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
else
    echo "==> Preparing classification kernel push..."
fi

# Push the kernel
echo "==> Pushing kernel to Kaggle..."
kaggle kernels push -p "$KAGGLE_DIR"

# Restore backup if detection mode
if [[ "$MODE" == "detection" ]] && [[ -f "$KAGGLE_DIR/kernel-metadata.json.bak" ]]; then
    mv "$KAGGLE_DIR/kernel-metadata.json.bak" "$KAGGLE_DIR/kernel-metadata.json"
fi

# Extract kernel slug from metadata
KERNEL_SLUG=$(python3 -c "
import json
with open('$KAGGLE_DIR/kernel-metadata.json') as f:
    meta = json.load(f)
slug = meta['id']
if '$MODE' == 'detection':
    slug = slug.replace('aerial-classification-training', 'aerial-detection-training')
print(slug)
")

echo "==> Kernel pushed: $KERNEL_SLUG"
echo "==> Monitoring status (Ctrl+C to stop monitoring)..."

# Poll status every 60 seconds
while true; do
    STATUS=$(kaggle kernels status "$KERNEL_SLUG" 2>&1 || true)
    TIMESTAMP=$(date '+%H:%M:%S')
    echo "[$TIMESTAMP] $STATUS"

    if echo "$STATUS" | grep -qi "complete"; then
        echo "==> Kernel COMPLETED!"
        echo "==> Downloading output..."
        kaggle kernels output "$KERNEL_SLUG" -p "$PROJECT_ROOT/models/${MODE}_kaggle"
        echo "==> Output saved to: $PROJECT_ROOT/models/${MODE}_kaggle"
        break
    elif echo "$STATUS" | grep -qi "error\|cancel"; then
        echo "==> Kernel FAILED or was CANCELLED."
        echo "==> Check logs: kaggle kernels output $KERNEL_SLUG -p /tmp/kaggle_logs"
        break
    fi

    sleep 60
done
