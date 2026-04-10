#!/bin/bash
# ============================================================
# Reset error experiments to pending (all models)
# Usage: bash scripts/domain_lora/reset_errors.sh
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

python -c "
import sqlite3, os

for model in ['aasist', 'conformertcm']:
    db_path = f'out/domain_lora/{model}/domain_lora.db'
    if not os.path.exists(db_path):
        continue
    conn = sqlite3.connect(db_path)
    errors = conn.execute(\"SELECT id, domain, lr, r, alpha FROM experiments WHERE status='error'\").fetchall()
    if not errors:
        print(f'{model}: no errors')
    else:
        for e in errors:
            print(f'  Reset #{e[0]}: {model} D{e[1]} lr={e[2]} r={e[3]} alpha={e[4]}')
        conn.execute(\"UPDATE experiments SET status='pending', pid=NULL, start_time=NULL, end_time=NULL WHERE status='error'\")
        conn.commit()
        print(f'{model}: reset {len(errors)} → pending')
    conn.close()
"
