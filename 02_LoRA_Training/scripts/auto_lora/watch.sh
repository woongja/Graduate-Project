#!/bin/bash
# AutoResearch LoRA Dashboard (auto-refresh)
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"
python scripts/auto_lora/dashboard.py --watch
