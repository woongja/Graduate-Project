#!/bin/bash
# ============================================================
# Sequential Domain Finetuning (D0→D1→D2→...→D7)
#
# Each domain builds on the previous domain's best model.
#   pretrained → D0 → D1 → D2 → ... → D7
#
# Usage:
#   bash scripts/train_all_domains.sh <MODEL> [GPU] [BATCH_SIZE]
#
# MODEL: aasist | conformertcm
#
# Examples:
#   bash scripts/train_all_domains.sh aasist
#   bash scripts/train_all_domains.sh conformertcm MIG-xxx 16
# ============================================================

set -e

MODEL="${1:?Usage: bash scripts/train_all_domains.sh <MODEL> [GPU] [BATCH_SIZE]}"
GPU=${2:-"MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"}
BS=${3:-24}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

DOMAINS=(1 2 3 4 5 6 7)

DOMAIN_NAMES=(
    "D0:clean"
    "D1:bg_noise+music"
    "D2:auto_tune"
    "D3:bandpass"
    "D4:echo"
    "D5:pitch+stretch"
    "D6:gaussian"
    "D7:reverberation"
)

TOTAL=${#DOMAINS[@]}

echo "============================================================"
echo " Sequential Domain Finetuning (Cumulative)"
echo "  Model  : $MODEL"
echo "  Domains: D0→D1→D2→...→D7"
echo "  GPU    : $GPU | Batch: $BS"
echo "  Total  : $TOTAL experiments"
echo "============================================================"
echo ""

# First domain starts from pretrained
if [ "$MODEL" = "aasist" ]; then
    PREV_CKPT="pretrained/aasist.pth"
else
    PREV_CKPT="pretrained/conformertcm.pth"
fi

for D in "${DOMAINS[@]}"; do
    echo ""
    echo "============================================================"
    echo "# [$(($D+1))/$TOTAL] $MODEL - ${DOMAIN_NAMES[$D]}"
    echo "#   Input : $PREV_CKPT"
    echo "#   Output: out/${MODEL}_D${D}/best_model.pth"
    echo "============================================================"
    echo ""

    bash scripts/train.sh "$MODEL" "$D" "$GPU" "$BS" "$PREV_CKPT"

    # Next domain uses this domain's output
    PREV_CKPT="out/${MODEL}_D${D}/best_model.pth"

    echo ""
    echo "[OK] $MODEL D${D} complete → next input: $PREV_CKPT"
    echo ""
done

echo ""
echo "============================================================"
echo " $MODEL: All $TOTAL sequential experiments complete!"
echo "  Chain: pretrained → D0 → D1 → ... → D7"
echo "============================================================"
