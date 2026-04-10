"""
Compute domain-wise EER from existing NC score files.

Usage:
    python scripts/compute_domain_eer.py
    python scripts/compute_domain_eer.py --csv results/domain_eer.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from evaluate_metrics import compute_eer

DOMAIN_TO_AUGTYPES = {
    0: ["clean"],
    1: ["background_noise", "background_music"],
    2: ["auto_tune"],
    3: ["high_pass_filter", "low_pass_filter"],
    4: ["echo"],
    5: ["pitch_shift", "time_stretch"],
    6: ["gaussian_noise"],
    7: ["reverberation"],
}

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOMAIN_NAMES = {
    0: "clean",
    1: "bg_noise+music",
    2: "auto_tune",
    3: "bandpass",
    4: "echo",
    5: "pitch+stretch",
    6: "gaussian",
    7: "reverberation",
}

CONFIGS = [
    ("aasist", "base_nc"),
    ("aasist", "finetune_nc"),
    ("conformertcm", "base_nc"),
    ("conformertcm", "finetune_nc"),
]

DATASETS = {
    "asv19": {
        "score_file": "asv19_nc_scores.txt",
        "protocol": os.path.join(BASE_DIR, "protocols", "asv19_eval.txt"),
    },
    "df21": {
        "score_file": "df21_nc_scores.txt",
        "protocol": os.path.join(BASE_DIR, "protocols", "df21_eval.txt"),
    },
}


def compute_domain_eers(score_path, protocol_path):
    """Compute overall + domain-wise EER."""
    proto = pd.read_csv(protocol_path, sep=" ", header=None, names=["filepath", "label", "aug_type"])
    score = pd.read_csv(score_path, sep=" ", header=None, names=["utt", "spoof", "bonafide"])
    merged = pd.merge(proto, score, left_on="filepath", right_on="utt")

    results = {}

    # Overall
    bf = merged[merged["label"] == "bonafide"]["bonafide"].values
    sp = merged[merged["label"] == "spoof"]["bonafide"].values
    if len(bf) > 0 and len(sp) > 0:
        eer, _ = compute_eer(bf, sp)
        results["overall"] = eer * 100
    else:
        results["overall"] = None

    # Per domain
    for d_id in range(8):
        allowed = DOMAIN_TO_AUGTYPES[d_id]
        dm = merged[merged["aug_type"].isin(allowed)]
        bf_d = dm[dm["label"] == "bonafide"]["bonafide"].values
        sp_d = dm[dm["label"] == "spoof"]["bonafide"].values
        if len(bf_d) > 0 and len(sp_d) > 0:
            eer_d, _ = compute_eer(bf_d, sp_d)
            results[f"D{d_id}"] = eer_d * 100
        else:
            results[f"D{d_id}"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute domain-wise EER")
    parser.add_argument("--csv", type=str, default=None, help="Export to CSV")
    args = parser.parse_args()

    all_results = []

    for model, mode in CONFIGS:
        for ds_name, ds_info in DATASETS.items():
            result_dir = os.path.join(BASE_DIR, "results", f"{model}_{mode}")
            score_path = os.path.join(result_dir, ds_info["score_file"])

            if not os.path.exists(score_path):
                print(f"[SKIP] {score_path} not found")
                continue

            eers = compute_domain_eers(score_path, ds_info["protocol"])
            row = {"model": model, "mode": mode, "dataset": ds_name}
            row.update(eers)
            all_results.append(row)

    if not all_results:
        print("No results found.")
        return

    # Display
    if HAS_RICH:
        console = Console()
        table = Table(title="Domain-wise EER (%)", show_lines=True)
        table.add_column("Model", width=13)
        table.add_column("Mode", width=13)
        table.add_column("Dataset", width=7)
        table.add_column("Overall", width=8, justify="right")
        for d in range(8):
            table.add_column(f"D{d}", width=8, justify="right")

        for r in all_results:
            overall = f"{r['overall']:.2f}" if r["overall"] is not None else "-"
            d_vals = []
            for d in range(8):
                v = r.get(f"D{d}")
                d_vals.append(f"{v:.2f}" if v is not None else "-")
            table.add_row(r["model"], r["mode"], r["dataset"], overall, *d_vals)

        console.print(table)
    else:
        header = f"{'Model':<13} {'Mode':<13} {'DS':<7} {'Overall':>8}"
        for d in range(8):
            header += f" {'D'+str(d):>8}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            overall = f"{r['overall']:.2f}" if r["overall"] is not None else "-"
            line = f"{r['model']:<13} {r['mode']:<13} {r['dataset']:<7} {overall:>8}"
            for d in range(8):
                v = r.get(f"D{d}")
                line += f" {v:>8.2f}" if v is not None else f" {'-':>8}"
            print(line)

    # CSV export
    if args.csv:
        fieldnames = ["model", "mode", "dataset", "overall"] + [f"D{d}" for d in range(8)]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_results:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"\nExported to {args.csv}")


if __name__ == "__main__":
    main()
