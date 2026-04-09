#!/usr/bin/env python3
"""
Normalize ASVspoof2019 LA and DF21 original_meta files into unified format.

Output columns: file_path, speaker, utt, label1
Outputs under protocols/processing_meta/:
  ASVspoof2019_train_meta.csv
  ASVspoof2019_dev_meta.csv
  ASVspoof2019_eval_meta.csv
  DF21_eval_meta.csv
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "Datasets"

SRC_DIR = ROOT / "protocols" / "original_meta"
DST_DIR = ROOT / "protocols" / "processing_meta"
DST_DIR.mkdir(parents=True, exist_ok=True)

FIELDNAMES = ["file_path", "speaker", "utt", "label1"]


def write_csv(dst_path, rows):
    with open(dst_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written {len(rows)} rows → {dst_path}")


# ── ASVspoof2019 LA train / dev ────────────────────────────────────────────────
def normalize_asv2019_traindev():
    src = SRC_DIR / "ASVspoof2019_LA_train_dev.txt"
    train_base = DATASETS / "ASVspoof" / "ASVspoof2019" / "ASVspoof2019_LA_train" / "flac"
    dev_base   = DATASETS / "ASVspoof" / "ASVspoof2019" / "ASVspoof2019_LA_dev"   / "flac"
    train_rows, dev_rows = [], []
    with open(src) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            filename, split, label = parts
            stem = Path(filename).stem
            if split == "train":
                file_path = train_base / filename
                train_rows.append({"file_path": str(file_path), "speaker": stem, "utt": stem, "label1": label})
            elif split == "dev":
                file_path = dev_base / filename
                dev_rows.append({"file_path": str(file_path), "speaker": stem, "utt": stem, "label1": label})
    write_csv(DST_DIR / "ASVspoof2019_train_meta.csv", train_rows)
    write_csv(DST_DIR / "ASVspoof2019_dev_meta.csv",   dev_rows)


# ── ASVspoof2019 LA eval ───────────────────────────────────────────────────────
def normalize_asv2019_eval():
    src = SRC_DIR / "protocol_asv19_eval.txt"
    base = DATASETS / "ASVspoof" / "ASVspoof2019_eval" / "flac"
    rows = []
    with open(src) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            filepath, split, label = parts
            filename = Path(filepath).name   # strip "flac/" prefix if present
            stem = Path(filename).stem
            file_path = base / filename
            rows.append({"file_path": str(file_path), "speaker": stem, "utt": stem, "label1": label})
    write_csv(DST_DIR / "ASVspoof2019_eval_meta.csv", rows)


# ── DF21 eval ─────────────────────────────────────────────────────────────────
def normalize_df21():
    src = SRC_DIR / "protocol_df21.txt"
    base = DATASETS / "ASVspoof" / "DF21_eval" / "flac"
    rows = []
    with open(src) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            filename, split, label = parts
            stem = Path(filename).stem
            file_path = base / filename
            rows.append({"file_path": str(file_path), "speaker": stem, "utt": stem, "label1": label})
    write_csv(DST_DIR / "DF21_eval_meta.csv", rows)


if __name__ == "__main__":
    normalize_asv2019_traindev()
    normalize_asv2019_eval()
    normalize_df21()
    print("Done.")
