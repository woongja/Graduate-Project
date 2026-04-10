"""
Domain LoRA Dashboard - shows all models.

Usage:
    python scripts/domain_lora/dashboard.py
    python scripts/domain_lora/dashboard.py --watch
"""

import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RUN_ORDER

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOMAIN_NAMES = {
    1: "bg_noise+music", 2: "auto_tune", 3: "bandpass",
    4: "echo", 5: "pitch+stretch", 6: "gaussian", 7: "reverberation",
}

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def get_experiments(model):
    db_path = os.path.join(BASE_DIR, "out", "domain_lora", model, "domain_lora.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM experiments ORDER BY domain, lr, r, alpha").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def read_progress(save_dir):
    if not save_dir:
        return ""
    try:
        with open(os.path.join(save_dir, "progress.txt")) as f:
            parts = f.read().strip().split(",")
            if len(parts) >= 2:
                return f"{parts[0]} {parts[1]}"
    except FileNotFoundError:
        pass
    return "starting..."


def format_lr(lr):
    if lr == 1e-5: return "1e-5"
    if lr == 5e-5: return "5e-5"
    if lr == 1e-4: return "1e-4"
    return f"{lr:.0e}"


def show_model(model, console=None):
    experiments = get_experiments(model)
    if not experiments:
        if console and HAS_RICH:
            console.print(f"[dim]{model}: no experiments[/dim]")
        else:
            print(f"{model}: no experiments")
        return

    styles = {"pending": "dim", "running": "bold yellow", "done": "bold green", "error": "bold red"}

    if HAS_RICH and console:
        table = Table(title=f"Domain LoRA — {model}", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Domain", width=16)
        table.add_column("lr", width=8)
        table.add_column("r", width=4, justify="right")
        table.add_column("alpha", width=5, justify="right")
        table.add_column("Status", width=20)
        table.add_column("Val Loss", width=10, justify="right")
        table.add_column("ASV19 EER", width=10, justify="right")
        table.add_column("DF21 EER", width=10, justify="right")

        for e in experiments:
            status = e["status"]
            style = styles.get(status, "")
            val = f"{e['best_val_loss']:.4f}" if e["best_val_loss"] is not None else "-"
            asv = f"{e['eer_asv19']:.4f}%" if e["eer_asv19"] is not None else "-"
            df = f"{e['eer_df21']:.4f}%" if e["eer_df21"] is not None else "-"
            domain = f"D{e['domain']} ({DOMAIN_NAMES.get(e['domain'], '')})"

            if status == "running":
                status_display = f"[{style}]{read_progress(e.get('save_dir'))}[/{style}]"
            else:
                status_display = f"[{style}]{status}[/{style}]"

            table.add_row(str(e["id"]), domain, format_lr(e["lr"]),
                          str(e["r"]), str(e["alpha"]), status_display, val, asv, df)

        console.print(table)

        # Summary
        counts = {}
        for e in experiments:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        lines = [" | ".join(f"{s}: {c}" for s, c in counts.items())]

        done_exps = [e for e in experiments if e["status"] == "done"]
        if done_exps:
            best_asv = min(done_exps, key=lambda e: e["eer_asv19"] if e["eer_asv19"] is not None else 999)
            best_df = min(done_exps, key=lambda e: e["eer_df21"] if e["eer_df21"] is not None else 999)
            if best_asv["eer_asv19"] is not None:
                lines.append(f"Best ASV19: {best_asv['eer_asv19']:.4f}% (D{best_asv['domain']}, lr={format_lr(best_asv['lr'])}, r={best_asv['r']}, a={best_asv['alpha']})")
            if best_df["eer_df21"] is not None:
                lines.append(f"Best DF21:  {best_df['eer_df21']:.4f}% (D{best_df['domain']}, lr={format_lr(best_df['lr'])}, r={best_df['r']}, a={best_df['alpha']})")

        console.print(Panel("\n".join(lines), title=f"{model} Summary"))
    else:
        print(f"\n=== {model} ===")
        for e in experiments:
            val = f"{e['best_val_loss']:.4f}" if e['best_val_loss'] else "-"
            asv = f"{e['eer_asv19']:.4f}%" if e['eer_asv19'] else "-"
            df = f"{e['eer_df21']:.4f}%" if e['eer_df21'] else "-"
            print(f"#{e['id']} D{e['domain']} lr={format_lr(e['lr'])} r={e['r']} a={e['alpha']} {e['status']} val={val} asv={asv} df={df}")


def show_all():
    console = Console() if HAS_RICH else None
    for model in RUN_ORDER:
        show_model(model, console)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system("clear")
                show_all()
                print(f"\nLast updated: {time.strftime('%H:%M:%S')} — Ctrl+C to exit")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
    else:
        show_all()
