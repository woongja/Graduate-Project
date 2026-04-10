#!/bin/bash
# ============================================================
# Pipeline Eval - Start scheduler
#
# Usage:
#   bash scripts/start.sh
# ============================================================

# ── GPU 목록 (여기에 UUID 추가/수정) ──
GPUS=(
    "MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"
    "MIG-6e4275af-2db0-51f1-a601-7ad8a1002745"
    "MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494"
    "MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"
    "MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd"
)

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

GPU_LIST=$(IFS=,; echo "${GPUS[*]}")

echo "============================================================"
echo " Pipeline Eval Scheduler"
echo "  GPUs: ${#GPUS[@]}"
echo "  Dashboard: bash scripts/watch.sh"
echo "  Stop: Ctrl+C"
echo "============================================================"
echo ""

python scripts/scheduler.py --gpus "$GPU_LIST"
