#!/bin/bash
# ============================================================
# All Domains LoRA Training (D1~D7, independent)
#
# Each domain gets its own LoRA from pretrained (not cumulative)
#
# Usage:
#   bash scripts/train_all_lora.sh <MODEL> [GPU] [BATCH_SIZE]
#
# Examples:
#   bash scripts/train_all_lora.sh aasist
#   bash scripts/train_all_lora.sh conformertcm MIG-xxx 16
# ============================================================

set -e

MODEL="${1:?Usage: bash scripts/train_all_lora.sh <MODEL> [GPU] [BATCH_SIZE]}"
GPU=${2:-"MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"}
BS=${3:-24}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

DOMAINS=(1 2 3 4 5 6 7)

DOMAIN_NAMES=(
    ""
    "D1:bg_noise+music"
    "D2:auto_tune"
    "D3:bandpass"
    "D4:echo"
    "D5:pitch+stretch"
    "D6:gaussian"
    "D7:reverberation"
)

TOTAL=${#DOMAINS[@]}
COUNT=0

echo "============================================================"
echo " Independent LoRA Training (D1~D7)"
echo "  Model  : $MODEL"
echo "  Domains: D1~D7 (each from pretrained)"
echo "  GPU    : $GPU | Batch: $BS"
echo "  Total  : $TOTAL experiments"
echo "============================================================"
echo ""

for D in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "============================================================"
    echo "# [$COUNT/$TOTAL] $MODEL - ${DOMAIN_NAMES[$D]}"
    echo "#   Output: out/multi_lora/${MODEL}_lora_D${D}/best_model.pth"
    echo "============================================================"
    echo ""

    bash scripts/train_lora.sh "$MODEL" "$D" "$GPU" "$BS"

    echo ""
    echo "[OK] $MODEL LoRA D${D} complete."
    echo ""
done

echo ""
echo "============================================================"
echo " $MODEL: All $TOTAL LoRA experiments complete!"
echo "  Models: out/multi_lora/${MODEL}_lora_D1 ~ out/multi_lora/${MODEL}_lora_D7"
echo "============================================================"
