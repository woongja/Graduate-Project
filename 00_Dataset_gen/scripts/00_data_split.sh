#!/usr/bin/env bash
# Create train/dev/eval protocol CSVs from processing_meta files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[00_data_split] ROOT=$ROOT"

# Check processing_meta exists
META_DIR="$ROOT/protocols/processing_meta"
if [ ! -d "$META_DIR" ] || [ -z "$(ls -A "$META_DIR")" ]; then
  echo "processing_meta not found or empty: $META_DIR"
  echo "Run: python3 $ROOT/scripts/normalize_meta.py"
  exit 1
fi

if [ -z "${CONDA_EXE:-}" ]; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
fi

echo "Activating conda env: dataset"
conda activate dataset

echo "Running protocol split..."
python3 "$ROOT/scripts/create_protocols.py"

conda deactivate || true

echo "[00_data_split] done → protocols/train.csv, dev.csv, eval.csv"
