#!/usr/bin/env bash
# Run augmentations for all splits (train / dev / eval).
# Non-autotune augmentations run in conda env `dataset`.
# Autotune augmentations run in conda env `dataset_autotune`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTOCOL_DIR="$ROOT/protocols"
OUTPUT_BASE="$ROOT/Datasets/noise_dataset/NC/"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

for SPLIT in train dev eval; do
  PROTO="$PROTOCOL_DIR/${SPLIT}.csv"
  if [ ! -f "$PROTO" ]; then
    echo "Protocol not found, skipping: $PROTO"
    continue
  fi

  echo "=== [$SPLIT] non-autotune augmentations ==="
  conda activate dataset
  OMP_NUM_THREADS=8 python "$ROOT/scripts/run_augmentations.py" \
    --protocol "$PROTO" \
    --split    "$SPLIT" \
    --output-base "$OUTPUT_BASE"
  conda deactivate

  echo "=== [$SPLIT] autotune augmentations ==="
  conda activate dataset_autotune
  OMP_NUM_THREADS=8 python "$ROOT/scripts/run_augmentations.py" \
    --protocol    "$PROTO" \
    --split       "$SPLIT" \
    --output-base "$OUTPUT_BASE" \
    --only-autotune
  conda deactivate
done

echo "All augmentations finished. Outputs under: $OUTPUT_BASE"
