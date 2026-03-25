#!/usr/bin/env bash
# Run augmentations for ASVspoof2019 (train/dev/eval) and DF21 (eval) independently.
# Non-autotune augmentations run in conda env `dataset`.
# Autotune augmentations run in conda env `dataset_autotune`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
META_DIR="$ROOT/protocols/processing_meta"
OUTPUT_BASE="$ROOT/Datasets/noise_dataset/NC_DF/"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Format: "meta_csv_filename:split_name"
DATASETS=(
    "ASVspoof2019_train_meta.csv:asv19_train"
    "ASVspoof2019_dev_meta.csv:asv19_dev"
    "ASVspoof2019_eval_meta.csv:asv19_eval"
    "DF21_eval_meta.csv:df21_eval"
)

for entry in "${DATASETS[@]}"; do
    meta_csv="${entry%%:*}"
    split="${entry##*:}"
    PROTO="$META_DIR/$meta_csv"

    if [ ! -f "$PROTO" ]; then
        echo "Protocol not found, skipping: $PROTO"
        continue
    fi

    echo "=== [$split] non-autotune augmentations ==="
    conda activate dataset
    OMP_NUM_THREADS=8 python "$ROOT/scripts/run_augmentations.py" \
        --protocol    "$PROTO" \
        --split       "$split" \
        --output-base "$OUTPUT_BASE"
    conda deactivate

    echo "=== [$split] autotune augmentations ==="
    conda activate dataset_autotune
    OMP_NUM_THREADS=8 python "$ROOT/scripts/run_augmentations.py" \
        --protocol    "$PROTO" \
        --split       "$split" \
        --output-base "$OUTPUT_BASE" \
        --only-autotune
    conda deactivate
done

echo "All DF augmentations finished. Outputs under: $OUTPUT_BASE"
