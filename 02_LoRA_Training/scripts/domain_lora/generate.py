"""
Generate domain LoRA experiments for all models.

Usage:
    python scripts/domain_lora/generate.py          # generate experiments
    python scripts/domain_lora/generate.py --clean  # clean + regenerate
"""

import argparse
import os
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS, RUN_ORDER

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_db(model):
    db_dir = os.path.join(BASE_DIR, "out", "domain_lora", model)
    db_path = os.path.join(db_dir, "domain_lora.db")
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain INTEGER NOT NULL,
            lr REAL NOT NULL,
            r INTEGER NOT NULL,
            alpha INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            pid INTEGER,
            gpu TEXT,
            start_time TEXT,
            end_time TEXT,
            best_val_loss REAL,
            eer_asv19 REAL,
            eer_df21 REAL,
            save_dir TEXT,
            UNIQUE(domain, lr, r, alpha)
        )
    """)
    conn.commit()
    return conn


def clean(model):
    db_dir = os.path.join(BASE_DIR, "out", "domain_lora", model)
    db_path = os.path.join(db_dir, "domain_lora.db")
    print(f"[CLEAN] {model}...")
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"  Deleted: {db_path}")
    if os.path.exists(db_dir):
        for item in os.listdir(db_dir):
            path = os.path.join(db_dir, item)
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  Deleted: {path}")


def generate(model):
    cfg = MODELS[model]
    domains = cfg["domains"]
    hps = cfg["hps"]

    if not hps:
        print(f"[SKIP] {model}: no HPs configured")
        return

    conn = get_db(model)
    db_dir = os.path.join(BASE_DIR, "out", "domain_lora", model)
    count = 0
    for domain in domains:
        for hp in hps:
            lr, r, alpha = hp["lr"], hp["r"], hp["alpha"]
            save_dir = os.path.join(db_dir, f"D{domain}_lr{lr}_r{r}_a{alpha}")
            try:
                conn.execute(
                    "INSERT INTO experiments (domain, lr, r, alpha, save_dir) VALUES (?, ?, ?, ?, ?)",
                    (domain, lr, r, alpha, save_dir)
                )
                count += 1
            except sqlite3.IntegrityError:
                pass
    conn.commit()
    conn.close()

    total = len(domains) * len(hps)
    print(f"[GENERATE] {model}: {count} new ({total} total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    for model in RUN_ORDER:
        if model not in MODELS:
            continue
        if args.clean:
            clean(model)
        generate(model)
