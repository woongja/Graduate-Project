#!/bin/bash
# ============================================================
# HP Search - Watch dashboard (auto-refresh)
#
# Usage:
#   bash scripts/hp_search/watch.sh
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

python scripts/hp_search/dashboard.py --watch
