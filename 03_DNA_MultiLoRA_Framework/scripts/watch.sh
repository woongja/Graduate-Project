#!/bin/bash
# ============================================================
# Pipeline Eval - Watch dashboard (auto-refresh)
# Usage: bash scripts/watch.sh
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

python dashboard.py --watch
