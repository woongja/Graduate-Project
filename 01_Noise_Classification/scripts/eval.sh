#!/usr/bin/env bash
# 통합 평가 스크립트
# 사용법: bash eval.sh <config_name> [model_path]
# 예시:
#   bash eval.sh ssast_base_patch_400
#   bash eval.sh ssast_base_patch_400 ../out/ssast_base_patch_400/best_0.pth
#   bash eval.sh htsat

set -e
cd "$(dirname "$0")"

CONFIG_NAME="${1:?'사용법: bash eval.sh <config_name> [model_path]  (예: ssast_base_patch_400)'}"
CONFIG="../config/${CONFIG_NAME}.yaml"
MODEL_PATH="${2:-../out/${CONFIG_NAME}/best.pth}"
GPU="MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"

if [ ! -f "$CONFIG" ]; then
    echo "[오류] config 파일을 찾을 수 없습니다: $CONFIG"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[오류] 모델 파일을 찾을 수 없습니다: $MODEL_PATH"
    exit 1
fi

echo "=============================="
echo "Config     : $CONFIG"
echo "Model path : $MODEL_PATH"
echo "GPU        : $GPU"
echo "=============================="

# ── in-domain (TIMIT 제외: VCTK + KsponSpeech) ──────────────────────────────
echo ""
echo "[1/2] in-domain eval (TIMIT 제외)"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU python ../train.py \
    --config "$CONFIG" \
    --is_eval \
    --model_path "$MODEL_PATH" \
    --eval_protocol "../protocols/eval_protocol_no_timit.txt" \
    --save_results "../results/in-domain/${CONFIG_NAME}_eval.txt"

# ── out-of-domain (TIMIT만) ───────────────────────────────────────────────────
echo ""
echo "[2/2] out-of-domain eval (TIMIT)"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU python ../train.py \
    --config "$CONFIG" \
    --is_eval \
    --model_path "$MODEL_PATH" \
    --eval_protocol "../protocols/eval_protocol_timit.txt" \
    --save_results "../results/out-of-domain/${CONFIG_NAME}_eval.txt"

echo ""
echo "=============================="
echo "완료"
echo "  in-domain  : ../results/in-domain/${CONFIG_NAME}_eval.txt"
echo "  out-of-domain : ../results/out-of-domain/${CONFIG_NAME}_eval.txt"
echo "=============================="
