#!/bin/bash
# ============================================================
# Domain Finetuning Script (via main.py)
#
# Usage:
#   bash scripts/train.sh <MODEL> <DOMAIN> [GPU] [BATCH_SIZE] [CKPT_PATH]
#
# MODEL:   aasist | conformertcm
# DOMAIN:  0~7
# CKPT_PATH: checkpoint to finetune from (default: pretrained)
#
# Examples:
#   bash scripts/train.sh aasist 1
#   bash scripts/train.sh aasist 2 MIG-xxx 24 out/aasist_D1/best_model.pth
# ============================================================

set -e

MODEL="${1:?Usage: bash scripts/train.sh <MODEL> <DOMAIN> [GPU] [BATCH_SIZE] [CKPT_PATH]}"
DOMAIN="${2:?Usage: bash scripts/train.sh <MODEL> <DOMAIN> [GPU] [BATCH_SIZE] [CKPT_PATH]}"
GPU=${3:-"MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"}
BS=${4:-24}
CKPT_OVERRIDE=${5:-""}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# ── Config ──
CONFIG="config/${MODEL}_baseline.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi

# ── Checkpoint ──
if [ -n "$CKPT_OVERRIDE" ]; then
    CKPT="$CKPT_OVERRIDE"
else
    if [ "$MODEL" = "aasist" ]; then
        CKPT="pretrained/aasist.pth"
    else
        CKPT="pretrained/conformertcm.pth"
    fi
fi

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    exit 1
fi

# ── Protocols ──
TRAIN_PROTO="protocols/asv19_train.txt"
DEV_PROTO="protocols/asv19_dev.txt"

# ── Save dir ──
SAVE_DIR="out/${MODEL}_D${DOMAIN}"

# ── GPU handling ──
export CUDA_VISIBLE_DEVICES="$GPU"

echo "============================================================"
echo " Domain Finetuning"
echo "  Model   : $MODEL"
echo "  Domain  : D${DOMAIN}"
echo "  Config  : $CONFIG"
echo "  Ckpt    : $CKPT"
echo "  Train   : $TRAIN_PROTO"
echo "  Dev     : $DEV_PROTO"
echo "  Save    : $SAVE_DIR"
echo "  GPU     : $GPU | Batch: $BS"
echo "============================================================"

python main.py \
    --config "$CONFIG" \
    --model_path "$CKPT" \
    --train_protocol "$TRAIN_PROTO" \
    --dev_protocol "$DEV_PROTO" \
    --domain "$DOMAIN" \
    --batch_size "$BS" \
    --save_dir "$SAVE_DIR"

echo ""
echo "============================================================"
echo " Training Complete!"
echo "  Model saved: ${SAVE_DIR}/best_model.pth"
echo "============================================================"
