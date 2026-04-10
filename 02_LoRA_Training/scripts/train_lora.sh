#!/bin/bash
# ============================================================
# Single Domain LoRA Training (via main.py)
#
# Usage:
#   bash scripts/train_lora.sh <MODEL> <DOMAIN> [GPU] [BATCH_SIZE]
#
# Each domain starts from pretrained (independent LoRA per domain)
# Config: config/{MODEL}_multi_lora_g{DOMAIN}.yaml
# Save:   out/multi_lora/{MODEL}/D{DOMAIN}_best.pth
# Logs:   out/multi_lora/logs/{MODEL}_D{DOMAIN}/
#
# Examples:
#   bash scripts/train_lora.sh aasist 1
#   bash scripts/train_lora.sh conformertcm 3 MIG-xxx 16
# ============================================================

set -e

MODEL="${1:?Usage: bash scripts/train_lora.sh <MODEL> <DOMAIN> [GPU] [BATCH_SIZE]}"
DOMAIN="${2:?Usage: bash scripts/train_lora.sh <MODEL> <DOMAIN> [GPU] [BATCH_SIZE]}"
GPU=${3:-"MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"}
BS=${4:-24}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# ── Config ──
CONFIG="config/${MODEL}_multi_lora_g${DOMAIN}.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi

# ── Always start from pretrained ──
if [ "$MODEL" = "aasist" ]; then
    CKPT="pretrained/aasist.pth"
else
    CKPT="pretrained/conformertcm.pth"
fi

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Pretrained checkpoint not found: $CKPT"
    exit 1
fi

# ── Protocols ──
TRAIN_PROTO="protocols/asv19_train.txt"
DEV_PROTO="protocols/asv19_dev.txt"

# ── Save & Log dirs ──
SAVE_DIR="out/multi_lora/${MODEL}"
LOG_DIR="out/multi_lora/logs/${MODEL}_D${DOMAIN}"

# ── GPU handling ──
export CUDA_VISIBLE_DEVICES="$GPU"

echo "============================================================"
echo " LoRA Domain Training"
echo "  Model   : $MODEL"
echo "  Domain  : D${DOMAIN}"
echo "  Config  : $CONFIG"
echo "  Ckpt    : $CKPT (pretrained)"
echo "  Save    : ${SAVE_DIR}/D${DOMAIN}_best.pth"
echo "  Logs    : $LOG_DIR"
echo "  GPU     : $GPU | Batch: $BS"
echo "============================================================"

python main.py \
    --config "$CONFIG" \
    --model_path "$CKPT" \
    --train_protocol "$TRAIN_PROTO" \
    --dev_protocol "$DEV_PROTO" \
    --domain "$DOMAIN" \
    --lora \
    --batch_size "$BS" \
    --save_dir "$SAVE_DIR" \
    --log_dir "$LOG_DIR"

echo ""
echo "============================================================"
echo " LoRA Training Complete!"
echo "  Model: ${SAVE_DIR}/D${DOMAIN}_best.pth"
echo "  Logs:  $LOG_DIR"
echo "============================================================"
