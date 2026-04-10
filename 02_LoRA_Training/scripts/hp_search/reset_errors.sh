#!/bin/bash
# ============================================================
# Reset error experiments to pending for retry
#
# Usage:
#   bash scripts/hp_search/reset_errors.sh
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

python -c "
import sqlite3
conn = sqlite3.connect('out/hp_search/hp_search.db')
errors = conn.execute(\"SELECT id, model, lr, r, alpha FROM experiments WHERE status='error'\").fetchall()
if not errors:
    print('No error experiments to reset.')
else:
    for e in errors:
        print(f'  Reset #{e[0]}: {e[1]} lr={e[2]} r={e[3]} alpha={e[4]}')
    conn.execute(\"UPDATE experiments SET status='pending', pid=NULL, start_time=NULL, end_time=NULL WHERE status='error'\")
    conn.commit()
    print(f'Reset {len(errors)} experiments → pending')
conn.close()
"
