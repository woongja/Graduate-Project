"""
Generate all NC × ADD × Dataset evaluation experiments.

Usage:
    python scripts/generate.py          # generate
    python scripts/generate.py --clean  # clean + regenerate
"""

import argparse
import os
import shutil
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "results", "eval_pipeline.db")

NC_MODULES = [
    "cnn8rnn_3ff_crossmodal",
    "cnn8rnn_3ff_base",
    "cnnlstm",
    "ssast_tiny",
    "htsat",
]
ADD_MODELS = ["aasist", "conformertcm"]
DATASETS = ["asv19", "df21", "itw"]


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nc TEXT NOT NULL,
            add_model TEXT NOT NULL,
            dataset TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            pid INTEGER,
            gpu TEXT,
            start_time TEXT,
            end_time TEXT,
            overall_eer REAL,
            UNIQUE(nc, add_model, dataset)
        )
    """)
    conn.commit()
    return conn


def clean():
    print("[CLEAN] Removing eval pipeline data...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Deleted: {DB_PATH}")
    results_dir = os.path.join(BASE_DIR, "results")
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            path = os.path.join(results_dir, item)
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  Deleted: {path}")
    print("[CLEAN] Done.")


def generate():
    conn = get_db()
    count = 0
    for nc in NC_MODULES:
        for add_model in ADD_MODELS:
            for ds in DATASETS:
                try:
                    conn.execute(
                        "INSERT INTO experiments (nc, add_model, dataset) VALUES (?, ?, ?)",
                        (nc, add_model, ds)
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    pass
    conn.commit()
    conn.close()

    total = len(NC_MODULES) * len(ADD_MODELS) * len(DATASETS)
    print(f"[GENERATE] {count} new experiments ({total} total)")
    print(f"  NC: {len(NC_MODULES)} | ADD: {len(ADD_MODELS)} | DS: {len(DATASETS)}")
    print(f"  DB: {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean:
        clean()
    generate()
