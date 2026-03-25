#!/usr/bin/env bash
# Run augmentation pipeline for ASVspoof2019 + DF21 datasets.
# Prerequisites: run normalize_meta_DF.py first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
META_DIR="$ROOT/protocols/processing_meta"

echo "[01_data_generate_DF] ROOT=$ROOT"

# Check processing_meta files exist
for f in ASVspoof2019_train_meta.csv ASVspoof2019_dev_meta.csv ASVspoof2019_eval_meta.csv DF21_eval_meta.csv; do
    if [ ! -f "$META_DIR/$f" ]; then
        echo "Meta file not found: $META_DIR/$f"
        echo "Run: python3 $ROOT/scripts/normalize_meta_DF.py"
        exit 1
    fi
done

bash "$ROOT/scripts/run_all_augmentations_DF.sh"

echo "[01_data_generate_DF] done"
