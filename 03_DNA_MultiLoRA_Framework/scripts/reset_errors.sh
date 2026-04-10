#!/bin/bash
# ============================================================
# Reset error experiments to pending
# Usage: bash scripts/reset_errors.sh
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

python -c "
import sqlite3, os
db_path = 'results/eval_pipeline.db'
if not os.path.exists(db_path):
    print('No DB found')
    exit()
conn = sqlite3.connect(db_path)
errors = conn.execute(\"SELECT id, nc, add_model, dataset FROM experiments WHERE status='error'\").fetchall()
if not errors:
    print('No error experiments')
else:
    for e in errors:
        print(f'  Reset #{e[0]}: {e[1]} + {e[2]} + {e[3]}')
    conn.execute(\"UPDATE experiments SET status='pending', pid=NULL, start_time=NULL, end_time=NULL WHERE status='error'\")
    conn.commit()
    print(f'Reset {len(errors)} → pending')
conn.close()
"
