#!/usr/bin/env bash
# SSAST Noise Classification Training
# Usage: bash scripts/train_ssast.sh [config_name]
#   config_name: ssast_base_patch_400 | ssast_small_patch_400 | ssast_tiny_patch_400
#   default: ssast_base_patch_400

CONFIG_NAME="${1:-ssast_base_patch_400}"
CONFIG="../config/${CONFIG_NAME}.yaml"
GPU="MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU python ../train.py \
    --config "$CONFIG" \
    --is_train
