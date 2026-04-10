#!/bin/bash
# ============================================================
# LoRA Model Evaluation (All Domains)
#
# For each LoRA (D1~D7):
#   - Full dataset inference (asv19, df21, itw)
#   - EER: full + domain-wise (from same score file)
#
# Usage:
#   bash scripts/eval_all_lora.sh <MODEL> [GPU] [BATCH_SIZE]
#
# Examples:
#   bash scripts/eval_all_lora.sh conformertcm
#   bash scripts/eval_all_lora.sh aasist MIG-xxx 64
# ============================================================

set -e

MODEL="${1:?Usage: bash scripts/eval_all_lora.sh <MODEL> [GPU] [BATCH_SIZE]}"
GPU=${2:-"MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"}
BS=${3:-64}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"

DOMAINS=(1 2 3 4 5 6 7)
DOMAIN_NAMES=("" "bg_noise+music" "auto_tune" "bandpass" "echo" "pitch+stretch" "gaussian" "reverberation")

# Original protocols (full dataset eval) — format: key subset label
PROTO_ASV19="protocols/original/protocol_asv19_eval.txt"
PROTO_DF21="protocols/original/protocol_df21.txt"
PROTO_ITW="protocols/original/protocol_itw.txt"

DB_ASV19="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_eval"
DB_DF21="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/DF21_eval/flac"
DB_ITW="/home/woongjae/ADD_LAB/Datasets/itw"

# Domain protocols — format: absolute_path label aug_type
PROTO_ASV19_DOMAIN="protocols/asv19_eval.txt"
PROTO_DF21_DOMAIN="protocols/df21_eval.txt"

RESULTS_DIR="results/multi_lora/${MODEL}"
mkdir -p "$RESULTS_DIR"

TOTAL_EVALS=0

# ── Compute full EER + domain-wise EER from score file ──
compute_eer_all() {
    local PROTO="$1"       # original protocol (key subset label)
    local SCORE="$2"       # score file
    local DOMAIN_PROTO="$3" # domain protocol (abs_path label aug_type), empty for ITW
    local LORA_D="$4"      # lora domain number
    local DATASET="$5"     # asv19 / df21 / itw

    python -c "
import pandas as pd, sys, os
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer
from show_results import ResultsDB
from datautils.data_utils import DOMAIN_TO_AUGTYPES

proto = pd.read_csv('$PROTO', sep=' ', header=None, names=['filepath','label','aug_type'])
score = pd.read_csv('$SCORE', sep=' ', header=None, names=['utt','spoof','bonafide'])

# Match by filepath (score utt = absolute path from nc_eval)
merged = pd.merge(proto, score, left_on='filepath', right_on='utt')

db = ResultsDB()

# ── Full EER ──
bf = merged[merged['label']=='bonafide']['bonafide'].values
sp = merged[merged['label']=='spoof']['bonafide'].values
eer, thr = compute_eer(bf, sp)
db.upsert('$MODEL', 'lora_D$LORA_D', '$DATASET', eer*100, thr)
print(f'  [Full] EER: {eer*100:.4f}% | bf={len(bf)} sp={len(sp)}')

# ── Domain-wise EER ──
domain_proto_path = '$DOMAIN_PROTO'
if domain_proto_path:
    for d_id in range(1, 8):
        allowed = DOMAIN_TO_AUGTYPES[d_id]
        dm = merged[merged['aug_type'].isin(allowed)]
        bf_d = dm[dm['label']=='bonafide']['bonafide'].values
        sp_d = dm[dm['label']=='spoof']['bonafide'].values

        if len(bf_d) == 0 or len(sp_d) == 0:
            continue

        eer_d, thr_d = compute_eer(bf_d, sp_d)
        db.upsert('$MODEL', 'lora_D${LORA_D}_domainD' + str(d_id), '$DATASET', eer_d*100, thr_d)
        print(f'  [D{d_id}] EER: {eer_d*100:.4f}% | bf={len(bf_d)} sp={len(sp_d)}')
"
}

echo "============================================================"
echo " LoRA Evaluation: $MODEL"
echo "  Domains: D1~D7"
echo "  GPU    : $GPU | Batch: $BS"
echo "============================================================"
echo ""

for D in "${DOMAINS[@]}"; do
    CKPT="out/multi_lora/${MODEL}/D${D}_best.pth"
    LORA_CONFIG="config/${MODEL}_multi_lora_g${D}.yaml"

    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] D${D}: checkpoint not found ($CKPT)"
        continue
    fi

    echo ""
    echo "############################################################"
    echo "# $MODEL LoRA D${D} (${DOMAIN_NAMES[$D]})"
    echo "############################################################"

    # ── ASV19 NC ──
    OUTPUT="${RESULTS_DIR}/D${D}_asv19.txt"
    echo ""
    echo "  ASV19 NC → $OUTPUT"
    python main.py --config "$LORA_CONFIG" --eval --nc_eval \
        --model_path "$CKPT" \
        --protocol_path "$PROTO_ASV19_DOMAIN" \
        --eval_output "$OUTPUT" \
        --batch_size "$BS"
    compute_eer_all "$PROTO_ASV19_DOMAIN" "$OUTPUT" "$PROTO_ASV19_DOMAIN" "$D" "asv19"
    TOTAL_EVALS=$((TOTAL_EVALS + 1))

    # ── DF21 NC ──
    OUTPUT="${RESULTS_DIR}/D${D}_df21.txt"
    echo ""
    echo "  DF21 NC → $OUTPUT"
    python main.py --config "$LORA_CONFIG" --eval --nc_eval \
        --model_path "$CKPT" \
        --protocol_path "$PROTO_DF21_DOMAIN" \
        --eval_output "$OUTPUT" \
        --batch_size "$BS"
    compute_eer_all "$PROTO_DF21_DOMAIN" "$OUTPUT" "$PROTO_DF21_DOMAIN" "$D" "df21"
    TOTAL_EVALS=$((TOTAL_EVALS + 1))

    # ── ITW (full only, no domain) ──
    OUTPUT="${RESULTS_DIR}/D${D}_itw.txt"
    echo ""
    echo "  ITW → $OUTPUT"
    python main.py --config "$LORA_CONFIG" --eval \
        --model_path "$CKPT" \
        --protocol_path "$PROTO_ITW" \
        --database_path "$DB_ITW" \
        --eval_output "$OUTPUT" \
        --batch_size "$BS"
    # ITW uses original protocol format (key subset label)
    python -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer
from show_results import ResultsDB
proto = pd.read_csv('$PROTO_ITW', sep=' ', header=None, names=['utt','subset','label'])
score = pd.read_csv('$OUTPUT', sep=' ', header=None, names=['utt','spoof','bonafide'])
merged = pd.merge(proto, score, on='utt')
bf = merged[merged['label']=='bonafide']['bonafide'].values
sp = merged[merged['label']=='spoof']['bonafide'].values
eer, thr = compute_eer(bf, sp)
db = ResultsDB()
db.upsert('$MODEL', 'lora_D$D', 'itw', eer*100, thr)
print(f'  [Full] EER: {eer*100:.4f}% | bf={len(bf)} sp={len(sp)}')
"
    TOTAL_EVALS=$((TOTAL_EVALS + 1))

    echo ""
    echo "[OK] $MODEL LoRA D${D} eval complete"
done

echo ""
echo "============================================================"
echo " $MODEL LoRA Evaluation Complete!"
echo "  Total: $TOTAL_EVALS inferences"
echo "  Results: $RESULTS_DIR"
echo "  Dashboard: python show_results.py"
echo "============================================================"
