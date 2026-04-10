#!/bin/bash
# ============================================================
# Run a single eval pipeline experiment
# Usage: bash scripts/run_single.sh <EXP_ID> <GPU>
# ============================================================

set -e

EXP_ID="${1:?Usage: bash scripts/run_single.sh <EXP_ID> <GPU>}"
GPU="${2:?Usage: bash scripts/run_single.sh <EXP_ID> <GPU>}"

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

export OMP_NUM_THREADS=4

DB_PATH="results/eval_pipeline.db"

# ── Read experiment params ──
read NC ADD_MODEL DATASET <<< $(python -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
row = conn.execute('SELECT nc, add_model, dataset FROM experiments WHERE id=?', ($EXP_ID,)).fetchone()
print(f'{row[0]} {row[1]} {row[2]}')
conn.close()
")

echo "============================================================"
echo " Pipeline Eval #${EXP_ID}"
echo "  NC=$NC  ADD=$ADD_MODEL  Dataset=$DATASET"
echo "  GPU=$GPU"
echo "============================================================"

# ── Update status ──
python -c "
import sqlite3
from datetime import datetime
conn = sqlite3.connect('$DB_PATH')
conn.execute('UPDATE experiments SET status=\"running\", start_time=?, pid=? WHERE id=?',
    (datetime.now().isoformat(), $$, $EXP_ID))
conn.commit()
conn.close()
"

# ── Run eval ──
python eval.py \
    --nc "$NC" \
    --add "$ADD_MODEL" \
    --dataset "$DATASET" \
    --gpu "$GPU"

# ── Read EER and update DB ──
python -c "
import sqlite3, os
from datetime import datetime

eer_path = 'results/${NC}__${ADD_MODEL}/${DATASET}_eer.txt'
overall_eer = None
if os.path.exists(eer_path):
    with open(eer_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == 'overall' and parts[1] != '-':
                overall_eer = float(parts[1])
                break

conn = sqlite3.connect('$DB_PATH')
conn.execute('''UPDATE experiments SET
    status='done', end_time=?, overall_eer=?
    WHERE id=?''',
    (datetime.now().isoformat(), overall_eer, $EXP_ID))
conn.commit()
conn.close()

print(f'Overall EER: {overall_eer:.4f}%' if overall_eer else 'EER: -')
"

echo ""
echo "[OK] Experiment #${EXP_ID} complete"
