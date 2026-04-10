#!/bin/bash
# ============================================================
# Evaluate Sequential Finetuned Models v2 (10-domain) — 병렬
#
# 4개 작업을 각각 다른 GPU에서 동시 실행:
#   1. aasist seq_D9
#   2. conformertcm seq_D9
#   3. xlsr_sls pretrained
#   4. xlsr_sls seq_D9
#
# Usage:
#   bash scripts/eval_all_models_v2.sh
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

BS=64

# ── GPU 할당 (작업당 1개) ──
GPU_AASIST="MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"
GPU_CONFORMER="MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494"
GPU_SLS_PRE="MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd"
GPU_SLS_SEQ="MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"

# Protocols
PROTO_ASV19_NC="protocols/asv19_eval.txt"
PROTO_DF21_NC="protocols/df21_eval.txt"
PROTO_ITW="protocols/original/protocol_itw.txt"
DB_ITW="/home/woongjae/ADD_LAB/Datasets/itw"

# ── Single job: eval one checkpoint on 3 datasets ──
run_job() {
    local MODEL="$1" CONFIG="$2" CKPT="$3" TAG="$4" GPU="$5"
    local RESULT_DIR="results/seq_v2/${MODEL}_${TAG}"
    local LOG="results/seq_v2/${MODEL}_${TAG}.log"
    mkdir -p "$RESULT_DIR"

    export CUDA_VISIBLE_DEVICES="$GPU"

    echo "[START] $MODEL / $TAG (GPU=$GPU)" | tee "$LOG"

    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] $CKPT not found" | tee -a "$LOG"
        return
    fi

    # 1. ASV19 NC
    echo "  [1/3] ASV19 NC..." | tee -a "$LOG"
    local OUTPUT="${RESULT_DIR}/asv19_nc_scores.txt"
    python main.py --config "$CONFIG" --eval --nc_eval \
        --model_path "$CKPT" \
        --protocol_path "$PROTO_ASV19_NC" \
        --eval_output "$OUTPUT" \
        --batch_size "$BS" >> "$LOG" 2>&1

    python -c "
import pandas as pd, os, sys
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer
proto = pd.read_csv('$PROTO_ASV19_NC', sep=' ', header=None, names=['filepath','label','aug_type'])
score = pd.read_csv('$OUTPUT', sep=' ', header=None, names=['utt','spoof','bonafide'])
proto['basename'] = proto['filepath'].apply(os.path.basename)
score['basename'] = score['utt'].apply(os.path.basename)
merged = pd.merge(proto, score, on='basename')
bf = merged[merged['label']=='bonafide']['bonafide'].values
sp = merged[merged['label']=='spoof']['bonafide'].values
eer, _ = compute_eer(bf, sp)
print(f'  ASV19 EER: {eer*100:.4f}%  (bf={len(bf)} sp={len(sp)})')
" 2>&1 | tee -a "$LOG"

    # 2. DF21 NC
    echo "  [2/3] DF21 NC..." | tee -a "$LOG"
    OUTPUT="${RESULT_DIR}/df21_nc_scores.txt"
    python main.py --config "$CONFIG" --eval --nc_eval \
        --model_path "$CKPT" \
        --protocol_path "$PROTO_DF21_NC" \
        --eval_output "$OUTPUT" \
        --batch_size "$BS" >> "$LOG" 2>&1

    python -c "
import pandas as pd, os, sys
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer
proto = pd.read_csv('$PROTO_DF21_NC', sep=' ', header=None, names=['filepath','label','aug_type'])
score = pd.read_csv('$OUTPUT', sep=' ', header=None, names=['utt','spoof','bonafide'])
proto['basename'] = proto['filepath'].apply(os.path.basename)
score['basename'] = score['utt'].apply(os.path.basename)
merged = pd.merge(proto, score, on='basename')
bf = merged[merged['label']=='bonafide']['bonafide'].values
sp = merged[merged['label']=='spoof']['bonafide'].values
eer, _ = compute_eer(bf, sp)
print(f'  DF21 EER:  {eer*100:.4f}%  (bf={len(bf)} sp={len(sp)})')
" 2>&1 | tee -a "$LOG"

    # 3. ITW
    echo "  [3/3] ITW..." | tee -a "$LOG"
    OUTPUT="${RESULT_DIR}/itw_scores.txt"
    python main.py --config "$CONFIG" --eval \
        --model_path "$CKPT" \
        --protocol_path "$PROTO_ITW" \
        --database_path "$DB_ITW" \
        --eval_output "$OUTPUT" \
        --batch_size "$BS" >> "$LOG" 2>&1

    python -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer
proto = pd.read_csv('$PROTO_ITW', sep=' ', header=None, names=['utt','subset','label'])
score = pd.read_csv('$OUTPUT', sep=' ', header=None, names=['utt','spoof','bonafide'])
merged = pd.merge(proto, score, on='utt')
bf = merged[merged['label']=='bonafide']['bonafide'].values
sp = merged[merged['label']=='spoof']['bonafide'].values
eer, _ = compute_eer(bf, sp)
print(f'  ITW EER:   {eer*100:.4f}%  (bf={len(bf)} sp={len(sp)})')
" 2>&1 | tee -a "$LOG"

    echo "[DONE] $MODEL / $TAG" | tee -a "$LOG"
}

echo "============================================================"
echo " Sequential Finetuned v2 Evaluation — Parallel"
echo "  4 jobs × 3 datasets each"
echo "============================================================"
echo ""

mkdir -p results/seq_v2

PIDS=()

# Job 1: aasist seq_D9
run_job "aasist" "config/aasist_baseline.yaml" "out/seq_v2/aasist/D9/D9_best.pth" "seq_D9" "$GPU_AASIST" &
PIDS+=($!)

# Job 2: conformertcm seq_D9
run_job "conformertcm" "config/conformertcm_baseline.yaml" "out/seq_v2/conformertcm/D9/D9_best.pth" "seq_D9" "$GPU_CONFORMER" &
PIDS+=($!)

# Job 3: xlsr_sls pretrained
run_job "xlsr_sls" "config/xlsr_sls_baseline.yaml" "pretrained/XLSR-SLS.pth" "pretrained" "$GPU_SLS_PRE" &
PIDS+=($!)

# Job 4: xlsr_sls seq_D9
run_job "xlsr_sls" "config/xlsr_sls_baseline.yaml" "out/seq_v2/xlsr_sls/D9/D9_best.pth" "seq_D9" "$GPU_SLS_SEQ" &
PIDS+=($!)

echo "  PIDs: ${PIDS[*]}"
echo "  Logs: results/seq_v2/*.log"
echo ""

# Wait + cleanup
trap "kill ${PIDS[*]} 2>/dev/null; exit" INT TERM
wait

echo ""
echo "============================================================"
echo " All evaluations complete!"
echo "  Results: results/seq_v2/"
echo "============================================================"
