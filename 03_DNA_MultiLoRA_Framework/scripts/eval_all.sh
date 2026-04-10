#!/bin/bash
# ============================================================
# DNA-MultiLoRA: Evaluate all NC × ADD combinations
#
# NC 5종 × ADD 2종 = 10 조합 × 2 데이터셋 = 20 평가
#
# Usage:
#   bash scripts/eval_all.sh [GPU]
#
# Examples:
#   bash scripts/eval_all.sh
#   bash scripts/eval_all.sh MIG-8cdeef83-092c-5a8d-a748-452f299e1df0
# ============================================================

set -e

GPU=${1:-"cuda:0"}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

NC_MODULES=(
    "cnn8rnn_3ff_crossmodal"
    "cnn8rnn_3ff_base"
    "cnnlstm"
    "ssast_tiny"
    "htsat"
)

ADD_MODELS=(
    "aasist"
    "conformertcm"
)

DATASETS=("asv19" "df21")

TOTAL=$((${#NC_MODULES[@]} * ${#ADD_MODELS[@]} * ${#DATASETS[@]}))
COUNT=0

echo "============================================================"
echo " DNA-MultiLoRA: Full Evaluation"
echo "  NC modules: ${#NC_MODULES[@]}"
echo "  ADD models: ${#ADD_MODELS[@]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Total: $TOTAL evaluations"
echo "  GPU: $GPU"
echo "============================================================"
echo ""

for NC in "${NC_MODULES[@]}"; do
    for ADD in "${ADD_MODELS[@]}"; do
        for DS in "${DATASETS[@]}"; do
            COUNT=$((COUNT + 1))
            echo ""
            echo "############################################################"
            echo "# [$COUNT/$TOTAL] NC=$NC | ADD=$ADD | DS=$DS"
            echo "############################################################"

            python eval.py \
                --nc "$NC" \
                --add "$ADD" \
                --dataset "$DS" \
                --gpu "$GPU"

            echo "[OK] $NC + $ADD + $DS complete"
        done
    done
done

echo ""
echo "============================================================"
echo " All $TOTAL evaluations complete!"
echo " Results: results/"
echo " Dashboard: python dashboard.py"
echo "============================================================"
