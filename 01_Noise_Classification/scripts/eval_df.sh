#!/usr/bin/env bash
# Deepfake 데이터셋 평가 스크립트 (ASV19 + DF21)
# 사용법: bash eval_df.sh <model_name> [gpu]
# 예시:
#   bash eval_df.sh cnnlstm
#   bash eval_df.sh cnn8rnn MIG-xxxxx
#   bash eval_df.sh cnn8rnn_3ff_softmax MIG-57de94a5-be15-5b5a-b67e-e118352d8a59

set -e
cd "$(dirname "$0")"

MODEL_NAME="${1:?'사용법: bash eval_df.sh <model_name> [gpu]  (예: cnnlstm, cnn8rnn)'}"
CONFIG="../config/${MODEL_NAME}.yaml"
MODEL_PATH="../out/${MODEL_NAME}/best.pth"
GPU="${2:-${GPU:-MIG-56c6e426-3d07-52cb-aa59-73892edacb69}}"

# Protocol paths
ASV19_EVAL_PROTOCOL="../../02_LoRA_Training/protocols/asv19_eval.txt"
DF21_EVAL_PROTOCOL="../../02_LoRA_Training/protocols/df21_eval.txt"

# Results directory
RESULTS_DIR="../results/deepfake"

# Validation
if [ ! -f "$CONFIG" ]; then
    echo "[오류] Config 파일을 찾을 수 없습니다: $CONFIG"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[오류] 모델 파일을 찾을 수 없습니다: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$ASV19_EVAL_PROTOCOL" ]; then
    echo "[오류] ASV19 eval protocol을 찾을 수 없습니다: $ASV19_EVAL_PROTOCOL"
    exit 1
fi

if [ ! -f "$DF21_EVAL_PROTOCOL" ]; then
    echo "[오류] DF21 eval protocol을 찾을 수 없습니다: $DF21_EVAL_PROTOCOL"
    exit 1
fi

echo "=============================================="
echo "Deepfake Dataset Evaluation"
echo "=============================================="
echo "Model name : $MODEL_NAME"
echo "Config     : $CONFIG"
echo "Model path : $MODEL_PATH"
echo "GPU        : $GPU"
echo "=============================================="
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# ── ASV19 Eval ────────────────────────────────────────────────────────────────
echo "[1/2] Evaluating on ASV19 eval set..."
echo "  Protocol: $ASV19_EVAL_PROTOCOL"
echo ""

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU python ../train.py \
    --config "$CONFIG" \
    --is_eval \
    --model_path "$MODEL_PATH" \
    --eval_protocol "$ASV19_EVAL_PROTOCOL" \
    --save_results "${RESULTS_DIR}/${MODEL_NAME}_asv19_eval.txt"

echo ""
echo "  ✓ Results saved: ${RESULTS_DIR}/${MODEL_NAME}_asv19_eval.txt"
echo ""

# ── DF21 Eval ─────────────────────────────────────────────────────────────────
echo "[2/2] Evaluating on DF21 eval set..."
echo "  Protocol: $DF21_EVAL_PROTOCOL"
echo ""

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU python ../train.py \
    --config "$CONFIG" \
    --is_eval \
    --model_path "$MODEL_PATH" \
    --eval_protocol "$DF21_EVAL_PROTOCOL" \
    --save_results "${RESULTS_DIR}/${MODEL_NAME}_df21_eval.txt"

echo ""
echo "  ✓ Results saved: ${RESULTS_DIR}/${MODEL_NAME}_df21_eval.txt"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "=============================================="
echo "Evaluation completed!"
echo "=============================================="
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo "  • ${MODEL_NAME}_asv19_eval.txt"
echo "  • ${MODEL_NAME}_df21_eval.txt"
echo ""
echo "To view results:"
echo "  cat ${RESULTS_DIR}/${MODEL_NAME}_asv19_eval.txt"
echo "  cat ${RESULTS_DIR}/${MODEL_NAME}_df21_eval.txt"
echo "=============================================="
