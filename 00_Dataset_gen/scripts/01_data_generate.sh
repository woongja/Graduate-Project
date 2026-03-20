#!/usr/bin/env bash
# Run the full augmentation generation pipeline (train / dev / eval).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[01_data_generate] ROOT=$ROOT"

# Check protocols exist
for SPLIT in train dev eval; do
  PROTO="$ROOT/protocols/${SPLIT}.csv"
  if [ ! -f "$PROTO" ]; then
    echo "Protocol file not found: $PROTO"
    echo "Run: bash $ROOT/scripts/00_data_split.sh"
    exit 1
  fi
done

echo "Running full augmentation pipeline (see run_all_augmentations.sh)"
bash "$ROOT/scripts/run_all_augmentations.sh"

echo "[01_data_generate] done"
