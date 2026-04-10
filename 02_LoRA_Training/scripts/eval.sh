#!/bin/bash
# ============================================================
# Evaluation Script (via main.py --eval)
#
# Usage:
#   bash scripts/eval.sh <MODEL> <MODE> <DATASET> [GPU] [BATCH_SIZE]
#
# MODEL:   aasist | conformertcm
# MODE:    base | finetune | single_lora | multi_lora_g0~g7
# DATASET: asv19 | df21 | itw
#
# Examples:
#   bash scripts/eval.sh aasist base itw
#   bash scripts/eval.sh aasist base asv19 MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494 64
#   bash scripts/eval.sh conformertcm finetune df21
#   bash scripts/eval.sh aasist single_lora asv19
# ============================================================

set -e

MODEL="${1:?Usage: bash scripts/eval.sh <MODEL> <MODE> <DATASET> [GPU] [BATCH_SIZE]}"
MODE="${2:?Usage: bash scripts/eval.sh <MODEL> <MODE> <DATASET> [GPU] [BATCH_SIZE]}"
DATASET="${3:?Usage: bash scripts/eval.sh <MODEL> <MODE> <DATASET> [GPU] [BATCH_SIZE]}"
GPU=${4:-"MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"}
BS=${5-64}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# ── Config file mapping ──
if [ "$MODE" = "base" ]; then
    CONFIG="config/${MODEL}_baseline.yaml"
else
    CONFIG="config/${MODEL}_${MODE}.yaml"
fi

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG"
    echo "Available: $(ls config/*.yaml | xargs -n1 basename | sed 's/.yaml//' | tr '\n' ' ')"
    exit 1
fi

# ── Checkpoint mapping ──
case "$MODE" in
    base)
        if [ "$MODEL" = "aasist" ]; then
            CKPT="pretrained/aasist.pth"
        elif [ "$MODEL" = "conformertcm" ]; then
            CKPT="pretrained/conformertcm.pth"
        elif [ "$MODEL" = "xlsr_sls" ]; then
            CKPT="pretrained/XLSR-SLS.pth"
        elif [ "$MODEL" = "xlsr_mamba" ]; then
            CKPT="pretrained/XLSR-Mamba-DF/model.safetensors"
        else
            CKPT="pretrained/${MODEL}.pth"
        fi
        ;;
    finetune)
        CKPT="out/${MODEL}_finetune/best_model.pth"
        ;;
    single_lora)
        CKPT="out/${MODEL}_single_lora/best_model.pth"
        ;;
    multi_lora_g[0-7])
        CKPT="out/${MODEL}_${MODE}/best_model.pth"
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo "Modes: base | finetune | single_lora | multi_lora_g0~g7"
        exit 1
        ;;
esac

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    exit 1
fi

# ── Protocol & database_path mapping ──
case "$DATASET" in
    asv19)
        PROTOCOL="protocols/original/protocol_asv19_eval.txt"
        DB_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_eval"
        ;;
    df21)
        PROTOCOL="protocols/original/protocol_df21.txt"
        DB_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/DF21_eval/flac"
        ;;
    itw)
        PROTOCOL="protocols/original/protocol_itw.txt"
        DB_PATH="/home/woongjae/ADD_LAB/Datasets/itw"
        ;;
    *)
        echo "[ERROR] Unknown dataset: $DATASET"
        echo "Datasets: asv19 | df21 | itw"
        exit 1
        ;;
esac

# ── Output path ──
OUTPUT="results/${MODEL}_${MODE}/${DATASET}_scores.txt"
mkdir -p "$(dirname "$OUTPUT")"

# ── GPU handling ──
if [[ "$GPU" == MIG-* ]] || [[ "$GPU" == GPU-* ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    DEVICE="cuda:0"
else
    DEVICE="$GPU"
fi

echo "============================================================"
echo " Evaluation"
echo "  Model   : $MODEL ($MODE)"
echo "  Dataset : $DATASET"
echo "  Config  : $CONFIG"
echo "  Ckpt    : $CKPT"
echo "  Protocol: $PROTOCOL"
echo "  DB Path : $DB_PATH"
echo "  Output  : $OUTPUT"
echo "  GPU     : $GPU | Batch: $BS"
echo "============================================================"

python main.py \
    --config "$CONFIG" \
    --eval \
    --model_path "$CKPT" \
    --protocol_path "$PROTOCOL" \
    --database_path "$DB_PATH" \
    --eval_output "$OUTPUT" \
    --batch_size "$BS"

# ── Compute EER & save to dashboard DB ──
python -c "
import pandas as pd, os, sys
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer
from show_results import ResultsDB

proto = pd.read_csv('$PROTOCOL', sep=' ', header=None, names=['utt','subset','label'])
score = pd.read_csv('$OUTPUT', sep=' ', header=None, names=['utt','spoof','bonafide'])
merged = pd.merge(proto, score, on='utt')

bf = merged[merged['label']=='bonafide']['bonafide'].values
sp = merged[merged['label']=='spoof']['bonafide'].values
eer, thr = compute_eer(bf, sp)

# Save to dashboard DB
db = ResultsDB()
db.upsert('$MODEL', '$MODE', '$DATASET', eer*100, thr)

print(f'')
print(f'==================================================')
print(f'Model: $MODEL ($MODE) | Dataset: $DATASET')
print(f'EER: {eer*100:.4f}% | Threshold: {thr:.4f}')
print(f'Bonafide: {len(bf)} | Spoof: {len(sp)}')
print(f'[Saved to dashboard DB]')
print(f'==================================================')
"

echo ""
echo "Scores: $OUTPUT"
echo "Done!"
