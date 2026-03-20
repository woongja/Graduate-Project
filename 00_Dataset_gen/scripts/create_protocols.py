#!/usr/bin/env python3
"""
Split processing_meta CSVs into unified train/dev/eval protocol files.

Rules:
  - Speaker-disjoint split at 5:2:3 ratio
  - LibriSpeech : no cap   (~100 utts/spk already)
  - VCTK        : cap 150 utts/spk
  - DSD         : cap  42 utts/spk
  - TIMIT       : eval only, no cap

Input  : protocols/processing_meta/{LibriSpeech,VCTK,DSD,TIMIT}_meta.csv
Output : protocols/train.csv  protocols/dev.csv  protocols/eval.csv
Columns: file_path, speaker, utt, label1, dataset
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_DIR  = ROOT / "protocols" / "processing_meta"
OUT_DIR   = ROOT / "protocols"

RANDOM_SEED = 42
FIELDNAMES  = ["file_path", "speaker", "utt", "label1", "dataset"]

# (meta_file, dataset_tag, utts_per_spk_cap, split_ratio)
# split_ratio None  → eval only
DATASET_CONFIG = [
    ("LibriSpeech_meta.csv", "LibriSpeech", None,  (5, 2, 3)),
    ("VCTK_meta.csv",        "VCTK",        150,   (5, 2, 3)),
    ("DSD_meta.csv",         "DSD",         42,    (5, 2, 3)),
    ("TIMIT_meta.csv",       "TIMIT",       None,  None),      # eval only
]


def load_rows(meta_file):
    with open(META_DIR / meta_file) as f:
        return list(csv.DictReader(f))


def sample_per_speaker(rows, cap, seed=RANDOM_SEED):
    """Group rows by speaker, sample up to `cap` per speaker (None = no cap)."""
    from collections import defaultdict
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    out = []
    for spk, spk_rows in by_spk.items():
        if cap and len(spk_rows) > cap:
            spk_rows = rng.sample(spk_rows, cap)
        out.extend(spk_rows)
    return out


def speaker_disjoint_split(rows, ratio=(5, 2, 3), seed=RANDOM_SEED):
    """Return (train_rows, dev_rows, eval_rows) with disjoint speaker sets."""
    rng = random.Random(seed)
    speakers = sorted({r["speaker"] for r in rows})
    rng.shuffle(speakers)

    total = sum(ratio)
    n = len(speakers)
    n_train = round(n * ratio[0] / total)
    n_dev   = round(n * ratio[1] / total)
    # eval gets the remainder
    train_spk = set(speakers[:n_train])
    dev_spk   = set(speakers[n_train:n_train + n_dev])
    eval_spk  = set(speakers[n_train + n_dev:])

    train = [r for r in rows if r["speaker"] in train_spk]
    dev   = [r for r in rows if r["speaker"] in dev_spk]
    eval_ = [r for r in rows if r["speaker"] in eval_spk]
    return train, dev, eval_


def process_dataset(meta_file, dataset_tag, cap, ratio):
    rows = load_rows(meta_file)
    rows = sample_per_speaker(rows, cap)

    # tag dataset column
    for r in rows:
        r["dataset"] = dataset_tag

    if ratio is None:
        # eval only
        return [], [], rows

    return speaker_disjoint_split(rows, ratio)


def write_split(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}: {len(rows)} rows")


def main():
    all_train, all_dev, all_eval = [], [], []

    for meta_file, dataset_tag, cap, ratio in DATASET_CONFIG:
        src = META_DIR / meta_file
        if not src.exists():
            print(f"[SKIP] {meta_file} not found")
            continue
        print(f"Processing {dataset_tag}...")
        train, dev, eval_ = process_dataset(meta_file, dataset_tag, cap, ratio)
        all_train.extend(train)
        all_dev.extend(dev)
        all_eval.extend(eval_)
        print(f"  train: {len(train)} utts ({len({r['speaker'] for r in train})} spk)")
        print(f"  dev  : {len(dev)} utts ({len({r['speaker'] for r in dev})} spk)")
        print(f"  eval : {len(eval_)} utts ({len({r['speaker'] for r in eval_})} spk)")

    print("\nWriting protocol files...")
    write_split(OUT_DIR / "train.csv", all_train)
    write_split(OUT_DIR / "dev.csv",   all_dev)
    write_split(OUT_DIR / "eval.csv",  all_eval)

    print("\nDone.")
    print(f"  Total train : {len(all_train)}")
    print(f"  Total dev   : {len(all_dev)}")
    print(f"  Total eval  : {len(all_eval)}")


if __name__ == "__main__":
    main()
