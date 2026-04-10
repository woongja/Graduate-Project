"""
Experiment Results Dashboard

Records and displays EER results from all evaluations.
Results are stored in results/dashboard.db (SQLite).

Usage:
    python show_results.py                       # show dashboard
    python show_results.py --export results.csv  # export CSV
    python show_results.py --add aasist base asv19 0.1499 1.6984  # manually add result
"""

import argparse
import csv
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dashboard.db")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


CATEGORIES = {
    "base": "Pretrained (Original)",
    "base_nc": "Pretrained (NC)",
    "finetune": "Seq. Finetuned (Original)",
    "finetune_nc": "Seq. Finetuned (NC)",
    "single_lora": "Single LoRA",
}
# multi_lora_g0~g7 handled dynamically


def get_category(mode):
    if mode in CATEGORIES:
        return CATEGORIES[mode]
    if mode.startswith("multi_lora_g"):
        return "Multi-LoRA (NC)"
    if "domainD" in mode:
        return "LoRA Domain-Matched (NC)"
    if mode.startswith("lora_D"):
        return "LoRA Full Eval (NC)"
    return mode


class ResultsDB:
    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                mode TEXT NOT NULL,
                dataset TEXT NOT NULL,
                eer REAL NOT NULL,
                threshold REAL,
                timestamp TEXT DEFAULT (datetime('now')),
                UNIQUE(model, mode, dataset)
            )
        """)
        self.conn.commit()

    def upsert(self, model, mode, dataset, eer, threshold=None):
        self.conn.execute("""
            INSERT INTO results (model, mode, dataset, eer, threshold, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model, mode, dataset)
            DO UPDATE SET eer=excluded.eer, threshold=excluded.threshold, timestamp=excluded.timestamp
        """, (model, mode, dataset, eer, threshold, datetime.now().isoformat()))
        self.conn.commit()

    def get_all(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM results ORDER BY model, mode, dataset"
        ).fetchall()]

    def get_pivot(self):
        """Pivot: each row = (model, mode), columns = (asv19, df21, itw)."""
        rows = self.get_all()
        pivot = {}
        for r in rows:
            key = (r["model"], r["mode"])
            if key not in pivot:
                pivot[key] = {"model": r["model"], "mode": r["mode"],
                              "asv19": None, "df21": None, "itw": None}
            pivot[key][r["dataset"]] = r["eer"]
        # Sort by category order then model
        order = ["base", "base_nc", "finetune", "finetune_nc", "single_lora"] + [f"multi_lora_g{i}" for i in range(8)] + [f"lora_D{i}" for i in range(1, 8)] + [f"lora_D{i}_domainD{i}" for i in range(1, 8)]
        model_order = ["aasist", "conformertcm"]
        def sort_key(item):
            mode = item["mode"]
            model = item["model"]
            idx = order.index(mode) if mode in order else 99
            midx = model_order.index(model) if model in model_order else 99
            # Group by category first, then model within category
            return (idx, midx)
        return sorted(pivot.values(), key=sort_key)


def show_rich(db):
    console = Console()
    pivot = db.get_pivot()

    if not pivot:
        console.print("[dim]No results yet. Run evaluations first.[/dim]")
        return

    table = Table(title="Experiment Results Dashboard", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Category", width=24)
    table.add_column("Model", style="bold", width=12)
    table.add_column("Mode", width=16)
    table.add_column("EER ASV19 (%)", width=14, justify="right")
    table.add_column("EER DF21 (%)", width=14, justify="right")
    table.add_column("EER ITW (%)", width=14, justify="right")

    prev_cat = None
    for i, r in enumerate(pivot, 1):
        cat = get_category(r["mode"])
        if cat == prev_cat:
            cat_display = ""
        else:
            cat_display = cat
            prev_cat = cat

        asv = f"{r['asv19']:.4f}" if r["asv19"] is not None else "-"
        df = f"{r['df21']:.4f}" if r["df21"] is not None else "-"
        itw = f"{r['itw']:.4f}" if r["itw"] is not None else "-"

        table.add_row(str(i), cat_display, r["model"], r["mode"], asv, df, itw)

    console.print(table)

    # Summary
    all_rows = db.get_all()
    lines = [f"Total evaluations: {len(all_rows)}"]
    for ds in ["asv19", "df21", "itw"]:
        ds_rows = [r for r in all_rows if r["dataset"] == ds]
        if ds_rows:
            best = min(ds_rows, key=lambda x: x["eer"])
            lines.append(f"Best {ds.upper()}: {best['eer']:.4f}% ({best['model']} / {best['mode']})")
    console.print(Panel("\n".join(lines), title="Summary"))


def show_plain(db):
    pivot = db.get_pivot()
    if not pivot:
        print("No results yet.")
        return

    print(f"\n{'#':<3} {'Category':<24} {'Model':<12} {'Mode':<16} {'ASV19':>10} {'DF21':>10} {'ITW':>10}")
    print("-" * 90)
    prev_cat = None
    for i, r in enumerate(pivot, 1):
        cat = get_category(r["mode"])
        cat_display = cat if cat != prev_cat else ""
        prev_cat = cat
        asv = f"{r['asv19']:.4f}" if r["asv19"] is not None else "-"
        df = f"{r['df21']:.4f}" if r["df21"] is not None else "-"
        itw = f"{r['itw']:.4f}" if r["itw"] is not None else "-"
        print(f"{i:<3} {cat_display:<24} {r['model']:<12} {r['mode']:<16} {asv:>10} {df:>10} {itw:>10}")


def export_csv(db, output):
    pivot = db.get_pivot()
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "mode", "category", "asv19", "df21", "itw"])
        w.writeheader()
        for r in pivot:
            w.writerow({
                "model": r["model"], "mode": r["mode"],
                "category": get_category(r["mode"]),
                "asv19": r["asv19"], "df21": r["df21"], "itw": r["itw"]
            })
    print(f"Exported to {output}")


def main():
    parser = argparse.ArgumentParser(description="Experiment Results Dashboard")
    parser.add_argument("--export", type=str, help="Export to CSV")
    parser.add_argument("--add", nargs=5, metavar=("MODEL", "MODE", "DATASET", "EER", "THRESHOLD"),
                        help="Add result: MODEL MODE DATASET EER THRESHOLD")
    args = parser.parse_args()

    db = ResultsDB()

    if args.add:
        model, mode, dataset, eer, thr = args.add
        db.upsert(model, mode, dataset, float(eer), float(thr))
        print(f"Added: {model}/{mode}/{dataset} EER={eer}%")
    elif args.export:
        export_csv(db, args.export)
    elif HAS_RICH:
        show_rich(db)
    else:
        show_plain(db)


if __name__ == "__main__":
    main()
