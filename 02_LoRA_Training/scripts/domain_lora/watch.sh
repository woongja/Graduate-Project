#!/bin/bash
# ============================================================
# Domain LoRA - Watch dashboard (auto-refresh)
# Usage: bash scripts/domain_lora/watch.sh
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

python scripts/domain_lora/dashboard.py --watch
