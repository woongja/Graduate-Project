#!/bin/bash

# 기본 설정
PROTOCOL_FILE="/home/woongjae/noise-tracing/multi-feature_fusion/ICASSP2026/protocols/NC_protocol.txt"
BATCH_SIZE=32
NUM_EPOCHS=100
LEARNING_RATE=1e-5
NUM_CLASSES=9
SAVE_PATH="out/icassp_v3.pth"
EARLY_STOP_PATIENCE=3
LOG_DIR="runs/icassp_preprocessed"  # TensorBoard 로그 디렉토리
input_height=128   # freq axis for spec
input_width=126    # time axis
f0_len=126      # F0 time frame 수
NUM_WORKERS=8
PREPROCESSED_DIR="/data/woongjae/preprocess/icassp/NC"

# 전처리 여부 확인
if [ ! -d "$PREPROCESSED_DIR" ]; then
    echo "전처리된 데이터가 없습니다. 전처리를 먼저 실행합니다..."
    python preprocess_features.py \
        --protocol_file $PROTOCOL_FILE \
        --output_dir $PREPROCESSED_DIR \
        --subset all \
    
    if [ $? -ne 0 ]; then
        echo "전처리 실패!"
        exit 1
    fi
fi

# 출력 디렉토리 생성
mkdir -p $(dirname $SAVE_PATH)

# Training 실행 (전처리된 데이터 사용)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --is_train \
    --preprocessed_dir $PREPROCESSED_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --num_classes $NUM_CLASSES \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --save_path $SAVE_PATH \
    --log_dir $LOG_DIR \
    --input_height $input_height \
    --input_width $input_width \
    --f0_len $f0_len \
    --num_workers $NUM_WORKERS