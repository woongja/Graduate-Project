#!/usr/bin/env bash
# AST Noise Classification Training

CONFIG="../config/ast.yaml"
GPU="MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"  # set to empty string for CPU

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU python ../train.py \
    --config "$CONFIG" \
    --is_train