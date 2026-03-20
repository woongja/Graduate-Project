#!/usr/bin/env python3
"""
Normalize original_meta CSVs into a unified format under protocols/processing_meta/.

Output columns: file_path, speaker, utt, label1
  - label1: 'spoof' for DSD, 'bonafide' for all others
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


# ── LibriSpeech ────────────────────────────────────────────────────────────────
def normalize_librispeech():
    src = SRC_DIR / "LibriSpeech_meta.csv"
    base = DATASETS / "LibriSpeech" / "wav16"
    rows = []
    with open(src) as f:
        for r in csv.DictReader(f):
            filename = Path(r["path"]).name          # e.g. 7241-90850-0013.wav
            file_path = base / filename
            rows.append({
                "file_path": str(file_path),
                "speaker":   r["Speaker name"].strip(),
                "utt":       r["utt"].strip(),
                "label1":    "bonafide",
            })
    write_csv(DST_DIR / "LibriSpeech_meta.csv", rows)


# ── VCTK ──────────────────────────────────────────────────────────────────────
def normalize_vctk():
    src = SRC_DIR / "VCTK_meta.csv"
    base = DATASETS / "VCTK" / "VCTK-Corpus" / "wav"
    rows = []
    with open(src) as f:
        for r in csv.DictReader(f):
            filename = Path(r["path"]).name          # e.g. p310_007.wav
            file_path = base / filename
            rows.append({
                "file_path": str(file_path),
                "speaker":   r["Speaker name"].strip(),
                "utt":       r["utt"].strip(),
                "label1":    "bonafide",
            })
    write_csv(DST_DIR / "VCTK_meta.csv", rows)


# ── TIMIT ─────────────────────────────────────────────────────────────────────
def normalize_timit():
    src = SRC_DIR / "TIMIT_meta.csv"
    old_base = "/nvme3/Datasets/WJ/TIMIT/"
    new_base = DATASETS / "TIMIT"
    rows = []
    with open(src) as f:
        for r in csv.DictReader(f):
            old_path = r["File_path"].strip()
            # strip old base prefix and remap to new base
            rel = old_path.replace(old_base, "")
            file_path = new_base / rel
            rows.append({
                "file_path": str(file_path),
                "speaker":   r["Speaker name"].strip(),
                "utt":       r["utt"].strip(),
                "label1":    "bonafide",
            })
    write_csv(DST_DIR / "TIMIT_meta.csv", rows)


# ── DSD ───────────────────────────────────────────────────────────────────────
def normalize_dsd():
    src = SRC_DIR / "DSD_meta_fake.csv"
    base = DATASETS / "2-DSD-corpus"
    rows = []
    with open(src) as f:
        for r in csv.DictReader(f):
            rel_path = r["path"].strip()             # e.g. Synthesizers/Elevenlabs/TTS_Sasha_12.wav
            file_path = base / rel_path
            rows.append({
                "file_path": str(file_path),
                "speaker":   r["Speaker name"].strip(),
                "utt":       r["utt"].strip(),
                "label1":    "spoof",
            })
    write_csv(DST_DIR / "DSD_meta.csv", rows)


if __name__ == "__main__":
    normalize_librispeech()
    normalize_vctk()
    normalize_timit()
    normalize_dsd()
    print("Done.")
