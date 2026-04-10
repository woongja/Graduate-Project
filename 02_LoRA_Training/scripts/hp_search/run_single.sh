#!/bin/bash
# ============================================================
# Run a single HP search experiment
# Usage: bash scripts/hp_search/run_single.sh <EXP_ID> <GPU>
# ============================================================

set -e

EXP_ID="${1:?Usage: bash scripts/hp_search/run_single.sh <EXP_ID> <GPU>}"
GPU="${2:?Usage: bash scripts/hp_search/run_single.sh <EXP_ID> <GPU>}"

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

DB_PATH="out/hp_search/hp_search.db"

# NC protocols for D1 eval
PROTO_ASV19_NC="protocols/asv19_eval.txt"
PROTO_DF21_NC="protocols/df21_eval.txt"

export CUDA_VISIBLE_DEVICES="$GPU"

# ── Read experiment params from DB ──
read MODEL LR R ALPHA SAVE_DIR <<< $(python -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
row = conn.execute('SELECT model, lr, r, alpha, save_dir FROM experiments WHERE id=?', ($EXP_ID,)).fetchone()
print(f'{row[0]} {row[1]} {row[2]} {row[3]} {row[4]}')
conn.close()
")

# ── Model-specific config & checkpoint ──
if [ "$MODEL" = "aasist" ]; then
    CONFIG="config/aasist_multi_lora_g1.yaml"
    PRETRAINED="pretrained/aasist.pth"
else
    CONFIG="config/conformertcm_multi_lora_g1.yaml"
    PRETRAINED="pretrained/conformertcm.pth"
fi

echo "============================================================"
echo " HP Search Experiment #${EXP_ID}"
echo "  model=$MODEL  lr=$LR  r=$R  alpha=$ALPHA"
echo "  Config: $CONFIG"
echo "  Save: $SAVE_DIR"
echo "  GPU: $GPU"
echo "============================================================"

# ── Update status to running ──
python -c "
import sqlite3
from datetime import datetime
conn = sqlite3.connect('$DB_PATH')
conn.execute('UPDATE experiments SET status=\"running\", start_time=?, pid=? WHERE id=?',
    (datetime.now().isoformat(), $$, $EXP_ID))
conn.commit()
conn.close()
"

# ── Train ──
python main.py \
    --config "$CONFIG" \
    --model_path "$PRETRAINED" \
    --train_protocol protocols/asv19_train.txt \
    --dev_protocol protocols/asv19_dev.txt \
    --domain 1 \
    --lora \
    --lora_r "$R" \
    --lora_alpha "$ALPHA" \
    --learning_rate "$LR" \
    --batch_size 32 \
    --max_epochs 100 \
    --patience 5 \
    --save_dir "$SAVE_DIR"

# ── Extract best val loss from training log ──
BEST_VAL_LOSS=$(python -c "
import csv
log_file = '$SAVE_DIR/logs/training.log'
try:
    with open(log_file) as f:
        reader = csv.DictReader(f)
        best = min(reader, key=lambda r: float(r['Val_Loss']))
        print(f\"{best['Val_Loss']}\")
except:
    print('999')
")

# ── Update best_val_loss ──
python -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
conn.execute('UPDATE experiments SET best_val_loss=? WHERE id=?', ($BEST_VAL_LOSS, $EXP_ID))
conn.commit()
conn.close()
"

# ── Find the saved model ──
CKPT="$SAVE_DIR/D1_best.pth"
if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    python -c "
import sqlite3
from datetime import datetime
conn = sqlite3.connect('$DB_PATH')
conn.execute('UPDATE experiments SET status=\"error\", end_time=? WHERE id=?',
    (datetime.now().isoformat(), $EXP_ID))
conn.commit()
conn.close()
"
    exit 1
fi

# ── Update progress for eval phase ──
echo "eval_asv19,−,−" > "$SAVE_DIR/progress.txt"

# ── Eval ASV19 D1 ──
ASV19_OUTPUT="$SAVE_DIR/asv19_d1_scores.txt"
python main.py \
    --config "$CONFIG" \
    --eval --nc_eval \
    --model_path "$CKPT" \
    --protocol_path "$PROTO_ASV19_NC" \
    --domain 1 \
    --lora_r "$R" \
    --lora_alpha "$ALPHA" \
    --eval_output "$ASV19_OUTPUT" \
    --batch_size 64

echo "eval_df21,−,−" > "$SAVE_DIR/progress.txt"

# ── Eval DF21 D1 ──
DF21_OUTPUT="$SAVE_DIR/df21_d1_scores.txt"
python main.py \
    --config "$CONFIG" \
    --eval --nc_eval \
    --model_path "$CKPT" \
    --protocol_path "$PROTO_DF21_NC" \
    --domain 1 \
    --lora_r "$R" \
    --lora_alpha "$ALPHA" \
    --eval_output "$DF21_OUTPUT" \
    --batch_size 64

# ── Compute EER & update DB ──
python -c "
import pandas as pd, os, sys
from datetime import datetime
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer
from datautils.data_utils import DOMAIN_TO_AUGTYPES
import sqlite3

allowed = DOMAIN_TO_AUGTYPES[1]

# ASV19 D1
proto_asv = pd.read_csv('$PROTO_ASV19_NC', sep=' ', header=None, names=['filepath','label','aug_type'])
proto_asv = proto_asv[proto_asv['aug_type'].isin(allowed)]
score_asv = pd.read_csv('$ASV19_OUTPUT', sep=' ', header=None, names=['utt','spoof','bonafide'])
merged_asv = pd.merge(proto_asv, score_asv, left_on='filepath', right_on='utt')
bf_asv = merged_asv[merged_asv['label']=='bonafide']['bonafide'].values
sp_asv = merged_asv[merged_asv['label']=='spoof']['bonafide'].values
eer_asv, _ = compute_eer(bf_asv, sp_asv)

# DF21 D1
proto_df = pd.read_csv('$PROTO_DF21_NC', sep=' ', header=None, names=['filepath','label','aug_type'])
proto_df = proto_df[proto_df['aug_type'].isin(allowed)]
score_df = pd.read_csv('$DF21_OUTPUT', sep=' ', header=None, names=['utt','spoof','bonafide'])
merged_df = pd.merge(proto_df, score_df, left_on='filepath', right_on='utt')
bf_df = merged_df[merged_df['label']=='bonafide']['bonafide'].values
sp_df = merged_df[merged_df['label']=='spoof']['bonafide'].values
eer_df, _ = compute_eer(bf_df, sp_df)

conn = sqlite3.connect('$DB_PATH')
conn.execute('''UPDATE experiments SET
    eer_asv19_d1=?, eer_df21_d1=?, status=\"done\", end_time=?
    WHERE id=?''',
    (eer_asv*100, eer_df*100, datetime.now().isoformat(), $EXP_ID))
conn.commit()
conn.close()

print(f'ASV19 D1 EER: {eer_asv*100:.4f}% | DF21 D1 EER: {eer_df*100:.4f}%')
"

echo ""
echo "============================================================"
echo " Experiment #${EXP_ID} Complete!"
echo "  model=$MODEL  lr=$LR  r=$R  alpha=$ALPHA"
echo "  Val Loss: $BEST_VAL_LOSS"
echo "============================================================"
