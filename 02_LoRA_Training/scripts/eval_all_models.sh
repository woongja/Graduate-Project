#!/bin/bash
# ============================================================
# Evaluate Pretrained + Sequential Finetuned Models
#
# For each model (aasist, conformertcm):
#   - Pretrained → Original (asv19, df21, itw) + NC (asv19, df21)
#   - Sequential finetuned (D7) → Original (asv19, df21, itw) + NC (asv19, df21)
#
# Usage:
#   bash scripts/eval_all_models.sh [GPU] [BATCH_SIZE]
# ============================================================

set -e

GPU=${1:-"MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"}
BS=${2:-64}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"

MODELS=("aasist" "conformertcm")

# Original protocols
PROTO_ASV19="protocols/original/protocol_asv19_eval.txt"
PROTO_DF21="protocols/original/protocol_df21.txt"
PROTO_ITW="protocols/original/protocol_itw.txt"
DB_ASV19="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_eval"
DB_DF21="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/DF21_eval/flac"
DB_ITW="/home/woongjae/ADD_LAB/Datasets/itw"

# NC protocols (absolute paths, no base_dir needed)
PROTO_ASV19_NC="protocols/asv19_eval.txt"
PROTO_DF21_NC="protocols/df21_eval.txt"

# ── EER for original protocol (key subset label) ──
compute_eer_orig() {
    local PROTO="$1" SCORE="$2" MODEL="$3" MODE="$4" DS="$5"
    python -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer
from show_results import ResultsDB
proto = pd.read_csv('$PROTO', sep=' ', header=None, names=['utt','subset','label'])
score = pd.read_csv('$SCORE', sep=' ', header=None, names=['utt','spoof','bonafide'])
merged = pd.merge(proto, score, on='utt')
bf = merged[merged['label']=='bonafide']['bonafide'].values
sp = merged[merged['label']=='spoof']['bonafide'].values
eer, thr = compute_eer(bf, sp)
db = ResultsDB()
db.upsert('$MODEL', '$MODE', '$DS', eer*100, thr)
print(f'  EER: {eer*100:.4f}% | bf={len(bf)} sp={len(sp)}')
"
}

# ── EER for NC protocol (abs_path label aug_type) ──
compute_eer_nc() {
    local PROTO="$1" SCORE="$2" MODEL="$3" MODE="$4" DS="$5"
    python -c "
import pandas as pd, os, sys
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer
from show_results import ResultsDB
proto = pd.read_csv('$PROTO', sep=' ', header=None, names=['filepath','label','aug_type'])
score = pd.read_csv('$SCORE', sep=' ', header=None, names=['utt','spoof','bonafide'])
# Match by basename
proto['basename'] = proto['filepath'].apply(os.path.basename)
score['basename'] = score['utt'].apply(os.path.basename)
merged = pd.merge(proto, score, on='basename')
bf = merged[merged['label']=='bonafide']['bonafide'].values
sp = merged[merged['label']=='spoof']['bonafide'].values
eer, thr = compute_eer(bf, sp)
db = ResultsDB()
db.upsert('$MODEL', '$MODE', '$DS', eer*100, thr)
print(f'  EER: {eer*100:.4f}% | bf={len(bf)} sp={len(sp)}')
"
}

# ── Run eval (inference + EER) ──
run_eval() {
    local CONFIG="$1" CKPT="$2" PROTO="$3" DB_PATH="$4" OUTPUT="$5"
    mkdir -p "$(dirname "$OUTPUT")"
    python main.py --config "$CONFIG" --eval \
        --model_path "$CKPT" \
        --protocol_path "$PROTO" \
        --database_path "$DB_PATH" \
        --eval_output "$OUTPUT" \
        --batch_size "$BS"
}

# ── Run NC eval (domain protocol, base_dir="") ──
run_eval_nc() {
    local CONFIG="$1" CKPT="$2" PROTO="$3" OUTPUT="$4"
    mkdir -p "$(dirname "$OUTPUT")"
    python main.py --config "$CONFIG" --eval --nc_eval \
        --model_path "$CKPT" \
        --protocol_path "$PROTO" \
        --eval_output "$OUTPUT" \
        --batch_size "$BS"
}

echo "============================================================"
echo " Pretrained + Finetuned Evaluation"
echo "  GPU: $GPU | Batch: $BS"
echo "============================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    CONFIG="config/${MODEL}_baseline.yaml"

    if [ "$MODEL" = "aasist" ]; then
        PRETRAINED="pretrained/aasist.pth"
    else
        PRETRAINED="pretrained/conformertcm.pth"
    fi

    echo ""
    echo "############################################################"
    echo "# $MODEL"
    echo "############################################################"

    # ── 1. Pretrained — NC datasets ──
    # echo ""
    # echo "--- Pretrained (NC) ---"

    # OUTPUT="results/${MODEL}_base_nc/asv19_nc_scores.txt"
    # run_eval_nc "$CONFIG" "$PRETRAINED" "$PROTO_ASV19_NC" "$OUTPUT"
    # compute_eer_nc "$PROTO_ASV19_NC" "$OUTPUT" "$MODEL" "base_nc" "asv19"

    # OUTPUT="results/${MODEL}_base_nc/df21_nc_scores.txt"
    # run_eval_nc "$CONFIG" "$PRETRAINED" "$PROTO_DF21_NC" "$OUTPUT"
    # compute_eer_nc "$PROTO_DF21_NC" "$OUTPUT" "$MODEL" "base_nc" "df21"

    # OUTPUT="results/${MODEL}_base_nc/itw_scores.txt"
    # run_eval "$CONFIG" "$PRETRAINED" "$PROTO_ITW" "$DB_ITW" "$OUTPUT"
    # compute_eer_orig "$PROTO_ITW" "$OUTPUT" "$MODEL" "base_nc" "itw"

    # ── 2. Sequential Finetuned — NC datasets ──
    CKPT="out/sequential_finetuning/${MODEL}_D7/best_model.pth"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "--- Sequential Finetuned (NC) ---"

        OUTPUT="results/${MODEL}_finetune_nc/asv19_nc_scores.txt"
        run_eval_nc "$CONFIG" "$CKPT" "$PROTO_ASV19_NC" "$OUTPUT"
        compute_eer_nc "$PROTO_ASV19_NC" "$OUTPUT" "$MODEL" "finetune_nc" "asv19"

        OUTPUT="results/${MODEL}_finetune_nc/df21_nc_scores.txt"
        run_eval_nc "$CONFIG" "$CKPT" "$PROTO_DF21_NC" "$OUTPUT"
        compute_eer_nc "$PROTO_DF21_NC" "$OUTPUT" "$MODEL" "finetune_nc" "df21"

        OUTPUT="results/${MODEL}_finetune_nc/itw_scores.txt"
        run_eval "$CONFIG" "$CKPT" "$PROTO_ITW" "$DB_ITW" "$OUTPUT"
        compute_eer_orig "$PROTO_ITW" "$OUTPUT" "$MODEL" "finetune_nc" "itw"
    else
        echo ""
        echo "[SKIP] Sequential finetuned: $CKPT not found"
    fi
done

echo ""
echo "============================================================"
echo " All evaluations complete!"
echo "  Dashboard: python show_results.py"
echo "============================================================"
