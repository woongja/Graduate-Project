#!/bin/bash
# ============================================================
# Run a single domain LoRA experiment
# Usage: bash scripts/domain_lora/run_single.sh <MODEL> <EXP_ID> <GPU>
# ============================================================

set -e

MODEL="${1:?Usage: bash scripts/domain_lora/run_single.sh <MODEL> <EXP_ID> <GPU>}"
EXP_ID="${2:?Usage: bash scripts/domain_lora/run_single.sh <MODEL> <EXP_ID> <GPU>}"
GPU="${3:?Usage: bash scripts/domain_lora/run_single.sh <MODEL> <EXP_ID> <GPU>}"

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

DB_PATH="out/domain_lora/${MODEL}/domain_lora.db"
PROTO_ASV19_NC="protocols/asv19_eval.txt"
PROTO_DF21_NC="protocols/df21_eval.txt"

export CUDA_VISIBLE_DEVICES="$GPU"

# ── Read experiment params from DB ──
read DOMAIN LR R ALPHA SAVE_DIR <<< $(python -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
row = conn.execute('SELECT domain, lr, r, alpha, save_dir FROM experiments WHERE id=?', ($EXP_ID,)).fetchone()
print(f'{row[0]} {row[1]} {row[2]} {row[3]} {row[4]}')
conn.close()
")

# ── Model-specific config & checkpoint ──
if [ "$MODEL" = "aasist" ]; then
    CONFIG="config/aasist_multi_lora_g${DOMAIN}.yaml"
    PRETRAINED="pretrained/aasist.pth"
else
    CONFIG="config/conformertcm_multi_lora_g${DOMAIN}.yaml"
    PRETRAINED="pretrained/conformertcm.pth"
fi

echo "============================================================"
echo " Domain LoRA Experiment #${EXP_ID}"
echo "  model=$MODEL  domain=D${DOMAIN}  lr=$LR  r=$R  alpha=$ALPHA"
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
    --domain "$DOMAIN" \
    --lora \
    --lora_r "$R" \
    --lora_alpha "$ALPHA" \
    --learning_rate "$LR" \
    --batch_size 32 \
    --max_epochs 100 \
    --patience 5 \
    --save_dir "$SAVE_DIR"

# ── Extract best val loss ──
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

python -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
conn.execute('UPDATE experiments SET best_val_loss=? WHERE id=?', ($BEST_VAL_LOSS, $EXP_ID))
conn.commit()
conn.close()
"

# ── Find checkpoint ──
CKPT="$SAVE_DIR/D${DOMAIN}_best.pth"
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

# ── Update progress ──
echo "eval_asv19,−,−" > "$SAVE_DIR/progress.txt"

# ── Eval ASV19 domain ──
ASV19_OUTPUT="$SAVE_DIR/asv19_d${DOMAIN}_scores.txt"
python main.py \
    --config "$CONFIG" \
    --eval --nc_eval \
    --model_path "$CKPT" \
    --protocol_path "$PROTO_ASV19_NC" \
    --domain "$DOMAIN" \
    --lora_r "$R" \
    --lora_alpha "$ALPHA" \
    --eval_output "$ASV19_OUTPUT" \
    --batch_size 64

echo "eval_df21,−,−" > "$SAVE_DIR/progress.txt"

# ── Eval DF21 domain ──
DF21_OUTPUT="$SAVE_DIR/df21_d${DOMAIN}_scores.txt"
python main.py \
    --config "$CONFIG" \
    --eval --nc_eval \
    --model_path "$CKPT" \
    --protocol_path "$PROTO_DF21_NC" \
    --domain "$DOMAIN" \
    --lora_r "$R" \
    --lora_alpha "$ALPHA" \
    --eval_output "$DF21_OUTPUT" \
    --batch_size 64

# ── Compute EER & update DB ──
python -c "
import pandas as pd, sys
from datetime import datetime
sys.path.insert(0, '.')
from evaluate_metrics import compute_eer

DOMAIN_TO_AUGTYPES = {
    0: ['clean'], 1: ['background_noise', 'background_music'],
    2: ['auto_tune'], 3: ['high_pass_filter', 'low_pass_filter'],
    4: ['echo'], 5: ['pitch_shift', 'time_stretch'],
    6: ['gaussian_noise'], 7: ['reverberation'],
}
allowed = DOMAIN_TO_AUGTYPES[$DOMAIN]
import sqlite3

def calc_eer(proto_path, score_path):
    proto = pd.read_csv(proto_path, sep=' ', header=None, names=['filepath','label','aug_type'])
    proto = proto[proto['aug_type'].isin(allowed)]
    score = pd.read_csv(score_path, sep=' ', header=None, names=['utt','spoof','bonafide'])
    merged = pd.merge(proto, score, left_on='filepath', right_on='utt')
    bf = merged[merged['label']=='bonafide']['bonafide'].values
    sp = merged[merged['label']=='spoof']['bonafide'].values
    eer, _ = compute_eer(bf, sp)
    return eer * 100

eer_asv = calc_eer('$PROTO_ASV19_NC', '$ASV19_OUTPUT')
eer_df = calc_eer('$PROTO_DF21_NC', '$DF21_OUTPUT')

conn = sqlite3.connect('$DB_PATH')
conn.execute('''UPDATE experiments SET
    eer_asv19=?, eer_df21=?, status='done', end_time=?
    WHERE id=?''',
    (eer_asv, eer_df, datetime.now().isoformat(), $EXP_ID))
conn.commit()
conn.close()

print(f'D$DOMAIN ASV19 EER: {eer_asv:.4f}% | DF21 EER: {eer_df:.4f}%')
"

echo ""
echo "============================================================"
echo " Experiment #${EXP_ID} Complete!"
echo "  model=$MODEL  D${DOMAIN}  lr=$LR  r=$R  alpha=$ALPHA"
echo "============================================================"
