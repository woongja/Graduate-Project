#!/bin/bash
# ============================================================
# Sequential Domain Finetuning v2 (D0→D1→...→D9)
#
# 새 도메인 매핑 (10개):
#   D0: clean
#   D1: background_noise
#   D2: background_music
#   D3: auto_tune
#   D4: bandpass (high+low)
#   D5: echo
#   D6: pitch_shift
#   D7: time_stretch
#   D8: gaussian_noise
#   D9: reverberation
#
# Usage:
#   bash scripts/train_all_domains_v2.sh <MODEL> [GPU] [BATCH_SIZE]
#
# MODEL: aasist | conformertcm | xlsr_sls
#
# Examples:
#   bash scripts/train_all_domains_v2.sh aasist
#   bash scripts/train_all_domains_v2.sh xlsr_sls MIG-xxx 16
# ============================================================

set -e

MODEL="${1:?Usage: bash scripts/train_all_domains_v2.sh <MODEL> [GPU] [BATCH_SIZE]}"
GPU=${2:-"MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"}
BS=${3:-24}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

DOMAINS=(1 2 3 4 5 6 7 8 9)  # D0(clean) 제외 — pretrained가 이미 clean 학습됨
DOMAIN_NAMES=(
    "D0:clean"
    "D1:background_noise"
    "D2:background_music"
    "D3:auto_tune"
    "D4:bandpass"
    "D5:echo"
    "D6:pitch_shift"
    "D7:time_stretch"
    "D8:gaussian_noise"
    "D9:reverberation"
)

TOTAL=${#DOMAINS[@]}

# ── Config ──
CONFIG="config/${MODEL}_baseline.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi

# ── Pretrained checkpoint ──
case "$MODEL" in
    aasist)        PRETRAINED="pretrained/aasist.pth" ;;
    conformertcm)  PRETRAINED="pretrained/conformertcm.pth" ;;
    xlsr_sls)      PRETRAINED="pretrained/XLSR-SLS.pth" ;;
    *)             echo "[ERROR] Unknown model: $MODEL"; exit 1 ;;
esac

if [ ! -f "$PRETRAINED" ]; then
    echo "[ERROR] Pretrained not found: $PRETRAINED"
    exit 1
fi

OUT_BASE="out/seq_v2"

echo "============================================================"
echo " Sequential Domain Finetuning v2 (10 domains)"
echo "  Model  : $MODEL"
echo "  Domains: D1→D2→...→D9 (D0 skip)"
echo "  GPU    : $GPU | Batch: $BS"
echo "  Output : $OUT_BASE/${MODEL}/"
echo "  Total  : $TOTAL experiments"
echo "============================================================"
echo ""

PREV_CKPT="$PRETRAINED"

for D in "${DOMAINS[@]}"; do
    SAVE_DIR="${OUT_BASE}/${MODEL}/D${D}"

    echo ""
    echo "============================================================"
    echo "# [$((D+1))/$TOTAL] $MODEL - ${DOMAIN_NAMES[$D]}"
    echo "#   Input : $PREV_CKPT"
    echo "#   Output: ${SAVE_DIR}/D${D}_best.pth"
    echo "============================================================"
    echo ""

    CUDA_VISIBLE_DEVICES="$GPU" python main.py \
        --config "$CONFIG" \
        --model_path "$PREV_CKPT" \
        --train_protocol protocols/asv19_train.txt \
        --dev_protocol protocols/asv19_dev.txt \
        --domain "$D" \
        --batch_size "$BS" \
        --save_dir "$SAVE_DIR" \
        --max_epochs 100 \
        --patience 5

    # Next domain uses this domain's checkpoint
    PREV_CKPT="${SAVE_DIR}/D${D}_best.pth"

    if [ ! -f "$PREV_CKPT" ]; then
        echo "[ERROR] Checkpoint not created: $PREV_CKPT"
        exit 1
    fi

    echo ""
    echo "[OK] $MODEL D${D} complete → next input: $PREV_CKPT"
    echo ""
done

echo ""
echo "============================================================"
echo " $MODEL: All $TOTAL sequential experiments complete!"
echo "  Chain: pretrained → D1 → D2 → ... → D9"
echo "  Output: $OUT_BASE/${MODEL}/"
echo "============================================================"
