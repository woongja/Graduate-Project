"""
Generate HP search experiments for D1 LoRA optimization.

Usage:
    python scripts/hp_search/generate_experiments.py          # generate 48 experiments
    python scripts/hp_search/generate_experiments.py --clean  # clean + regenerate
"""

import argparse
import os
import shutil
import sqlite3
from itertools import product

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "out", "hp_search", "hp_search.db")

MODELS = ["aasist", "conformertcm"]
LRS = [1e-5, 5e-5, 1e-4]
RS = [4, 8, 16, 32]
ALPHA_MULTIPLIERS = [1, 2]  # alpha = r * multiplier


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            lr REAL NOT NULL,
            r INTEGER NOT NULL,
            alpha INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            pid INTEGER,
            gpu TEXT,
            start_time TEXT,
            end_time TEXT,
            best_val_loss REAL,
            eer_asv19_d1 REAL,
            eer_df21_d1 REAL,
            save_dir TEXT,
            UNIQUE(model, lr, r, alpha)
        )
    """)
    conn.commit()
    return conn


def clean():
    print("[CLEAN] Removing existing HP search data...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Deleted: {DB_PATH}")

    hp_out = os.path.join(BASE_DIR, "out", "hp_search")
    if os.path.exists(hp_out):
        for item in os.listdir(hp_out):
            path = os.path.join(hp_out, item)
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  Deleted: {path}")

    print("[CLEAN] Done.")


def generate():
    conn = get_db()
    count = 0
    for model, lr, r, mult in product(MODELS, LRS, RS, ALPHA_MULTIPLIERS):
        alpha = r * mult
        save_dir = os.path.join(BASE_DIR, "out", "hp_search", f"{model}_lr{lr}_r{r}_a{alpha}")

        try:
            conn.execute(
                "INSERT INTO experiments (model, lr, r, alpha, save_dir) VALUES (?, ?, ?, ?, ?)",
                (model, lr, r, alpha, save_dir)
            )
            count += 1
        except sqlite3.IntegrityError:
            pass  # already exists

    conn.commit()
    conn.close()

    total = len(MODELS) * len(LRS) * len(RS) * len(ALPHA_MULTIPLIERS)
    print(f"[GENERATE] {count} new experiments added ({total} total combinations)")
    print(f"  Models: {MODELS}")
    print(f"  DB: {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Clean existing data before generating")
    args = parser.parse_args()

    if args.clean:
        clean()

    generate()
