#!/bin/bash
# ============================================================
# Domain LoRA - Start scheduler
# config.py에서 MODEL, GPU 설정 후 실행
#
# Usage:
#   bash scripts/domain_lora/start.sh
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

python scripts/domain_lora/scheduler.py
