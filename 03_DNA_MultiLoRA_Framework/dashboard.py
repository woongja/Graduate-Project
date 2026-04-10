"""
DNA-MultiLoRA Results Dashboard

Usage:
    python dashboard.py
    python dashboard.py --watch
    python dashboard.py --csv results/summary.csv
"""

import argparse
import csv
import os
import sqlite3
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DB_PATH = os.path.join(RESULTS_DIR, "eval_pipeline.db")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

DOMAIN_NAMES = {
    0: "clean", 1: "bg_noise+music", 2: "auto_tune", 3: "bandpass",
    4: "echo", 5: "pitch+stretch", 6: "gaussian", 7: "reverberation",
}


def get_experiments():
    """Read experiments from DB."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM experiments ORDER BY nc, add_model, dataset").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_eer_file(nc, add_model, dataset):
    """Load domain-wise EER from file."""
    eer_path = os.path.join(RESULTS_DIR, f"{nc}__{add_model}", f"{dataset}_eer.txt")
    if not os.path.exists(eer_path):
        return {}
    results = {}
    with open(eer_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                results[parts[0]] = float(parts[1]) if parts[1] != "-" else None
    return results


def show_dashboard(export_csv=None):
    experiments = get_experiments()

    if not experiments:
        print("No experiments. Run: python scripts/generate.py")
        return

    styles = {"pending": "dim", "running": "bold yellow", "done": "bold green", "error": "bold red"}

    if HAS_RICH:
        console = Console()

        # Main status table
        table = Table(title="DNA-MultiLoRA Pipeline Results", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("NC Module", width=25)
        table.add_column("ADD Model", width=13)
        table.add_column("Dataset", width=7)
        table.add_column("Status", width=10)
        table.add_column("Overall EER", width=12, justify="right")

        for e in experiments:
            status = e["status"]
            style = styles.get(status, "")
            eer = f"{e['overall_eer']:.4f}%" if e["overall_eer"] is not None else "-"
            table.add_row(
                str(e["id"]), e["nc"], e["add_model"], e["dataset"],
                f"[{style}]{status}[/{style}]", eer
            )

        console.print(table)

        # Summary
        counts = {}
        for e in experiments:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        lines = [" | ".join(f"{s}: {c}" for s, c in counts.items())]

        done_with_eer = [e for e in experiments if e["status"] == "done" and e["overall_eer"] is not None]
        if done_with_eer:
            for ds in ["asv19", "df21", "itw"]:
                ds_exps = [e for e in done_with_eer if e["dataset"] == ds]
                if ds_exps:
                    best = min(ds_exps, key=lambda e: e["overall_eer"])
                    lines.append(f"Best {ds.upper()}: {best['overall_eer']:.4f}% ({best['nc']} + {best['add_model']})")

        console.print(Panel("\n".join(lines), title="Summary"))

    else:
        print(f"\n{'#':<3} {'NC':<25} {'ADD':<13} {'DS':<7} {'Status':<10} {'EER':>12}")
        print("-" * 75)
        for e in experiments:
            eer = f"{e['overall_eer']:.4f}%" if e["overall_eer"] is not None else "-"
            print(f"{e['id']:<3} {e['nc']:<25} {e['add_model']:<13} {e['dataset']:<7} {e['status']:<10} {eer:>12}")

    # CSV export
    if export_csv:
        all_rows = []
        for e in experiments:
            row = {"nc": e["nc"], "add_model": e["add_model"], "dataset": e["dataset"],
                   "status": e["status"], "overall_eer": e["overall_eer"]}
            eer_data = load_eer_file(e["nc"], e["add_model"], e["dataset"])
            for d in range(8):
                row[f"D{d}"] = eer_data.get(f"D{d}")
            all_rows.append(row)

        fieldnames = ["nc", "add_model", "dataset", "status", "overall_eer"] + [f"D{d}" for d in range(8)]
        with open(export_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nExported to {export_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNA-MultiLoRA Dashboard")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh every 10s")
    parser.add_argument("--csv", type=str, default=None, help="Export to CSV")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system("clear")
                show_dashboard()
                print(f"\nLast updated: {time.strftime('%H:%M:%S')} — Ctrl+C to exit")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
    else:
        show_dashboard(export_csv=args.csv)
