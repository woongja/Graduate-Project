#!/bin/bash
# ============================================================
# HP Search - Start scheduler (multi-GPU, single process)
#
# GPU 목록을 아래에 직접 수정하세요.
# 1개 스케줄러가 모든 GPU를 순회하며 실험을 할당합니다.
#
# Usage:
#   bash scripts/hp_search/start.sh
#
# Stop: Ctrl+C
# ============================================================

# ── 사용할 GPU 목록 (여기에 UUID 추가/수정) ──
GPUS=(
    "MIG-6e4275af-2db0-51f1-a601-7ad8a1002745"
    "MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"
    "MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"
    "MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494"
)

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

# Convert array to comma-separated string
GPU_LIST=$(IFS=,; echo "${GPUS[*]}")

echo "============================================================"
echo " HP Search - ${#GPUS[@]} GPUs"
echo "  Dashboard: bash scripts/hp_search/watch.sh"
echo "  Stop: Ctrl+C"
echo "============================================================"
echo ""

python scripts/hp_search/scheduler.py --gpus "$GPU_LIST"
