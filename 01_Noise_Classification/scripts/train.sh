#!/usr/bin/env bash
# 통합 학습 스크립트
# 사용법: bash train.sh <config_name>
# 예시:
#   bash train.sh ssast_base_patch_400
#   bash train.sh ssast_small_patch_400
#   bash train.sh ssast_tiny_patch_400
#   bash train.sh htsat

set -e
cd "$(dirname "$0")"

CONFIG_NAME="${1:?'사용법: bash train.sh <config_name>  (예: ssast_base_patch_400)'}"
CONFIG="../config/${CONFIG_NAME}.yaml"
GPU="MIG-56c6e426-3d07-52cb-aa59-73892edacb69"

if [ ! -f "$CONFIG" ]; then
    echo "[오류] config 파일을 찾을 수 없습니다: $CONFIG"
    exit 1
fi

echo "=============================="
echo "Config : $CONFIG"
echo "GPU    : $GPU"
echo "=============================="

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU python ../train.py \
    --config "$CONFIG" \
    --is_train
