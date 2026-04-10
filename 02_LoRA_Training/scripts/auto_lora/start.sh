#!/bin/bash
# ============================================================
# AutoResearch LoRA — 3모델 × GPU 2개 병렬 HP 탐색
#
# 각 모델에 GPU 2개 할당 → trial 2개 동시 실행
# Optuna DB 공유로 trial 중복 방지
#
# Usage:
#   bash scripts/auto_lora/start.sh
# ============================================================

# ── 모델별 GPU 할당 (각 2개) ──
AASIST_GPU1="MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"
AASIST_GPU2="MIG-6e4275af-2db0-51f1-a601-7ad8a1002745"

CONFORMERTCM_GPU1="MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494"
CONFORMERTCM_GPU2="MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"

XLSR_SLS_GPU1="MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd"
XLSR_SLS_GPU2="GPU-589d86ff-f8c6-815f-a780-1afb008be925"

# ── 탐색 도메인 ──
DOMAIN=1

# ── trial 수 (각 프로세스당) ──
N_TRIALS=25

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

echo "============================================================"
echo " AutoResearch LoRA — 3 Models × 2 GPUs"
echo "  Domain: D${DOMAIN}"
echo "  Trials: ${N_TRIALS} per process (${N_TRIALS}×2 per model)"
echo "============================================================"
echo ""

PIDS=()

# ── AASIST (2 parallel trials) ──
python scripts/auto_lora/runner.py --model aasist --gpu "$AASIST_GPU1" --domain "$DOMAIN" --n_trials "$N_TRIALS" &
PIDS+=($!)
sleep 2  # DB 초기화 race condition 방지
python scripts/auto_lora/runner.py --model aasist --gpu "$AASIST_GPU2" --domain "$DOMAIN" --n_trials "$N_TRIALS" &
PIDS+=($!)
echo "  aasist: 2 workers on $AASIST_GPU1, $AASIST_GPU2"

# ── ConformerTCM (2 parallel trials) ──
python scripts/auto_lora/runner.py --model conformertcm --gpu "$CONFORMERTCM_GPU1" --domain "$DOMAIN" --n_trials "$N_TRIALS" &
PIDS+=($!)
sleep 2
python scripts/auto_lora/runner.py --model conformertcm --gpu "$CONFORMERTCM_GPU2" --domain "$DOMAIN" --n_trials "$N_TRIALS" &
PIDS+=($!)
echo "  conformertcm: 2 workers on $CONFORMERTCM_GPU1, $CONFORMERTCM_GPU2"

# ── XLSR-SLS (2 parallel trials) ──
python scripts/auto_lora/runner.py --model xlsr_sls --gpu "$XLSR_SLS_GPU1" --domain "$DOMAIN" --n_trials "$N_TRIALS" &
PIDS+=($!)
sleep 2
python scripts/auto_lora/runner.py --model xlsr_sls --gpu "$XLSR_SLS_GPU2" --domain "$DOMAIN" --n_trials "$N_TRIALS" &
PIDS+=($!)
echo "  xlsr_sls: 2 workers on $XLSR_SLS_GPU1, $XLSR_SLS_GPU2"

echo ""
echo "  Results: out/auto_lora/{model}/results.tsv"
echo "  Stop all: kill ${PIDS[*]}"
echo "============================================================"
echo ""

# Wait + cleanup on Ctrl+C
trap "kill ${PIDS[*]} 2>/dev/null; exit" INT TERM
wait
echo "All models complete!"
