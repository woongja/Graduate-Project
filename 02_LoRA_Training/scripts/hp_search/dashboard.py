"""
HP Search Dashboard - display experiment status and results.

Usage:
    python scripts/hp_search/dashboard.py
    python scripts/hp_search/dashboard.py --watch   # auto-refresh every 10s
"""

import argparse
import os
import sqlite3
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "out", "hp_search", "hp_search.db")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def get_experiments():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM experiments ORDER BY model, lr, r, alpha"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def read_progress(save_dir):
    """Read progress.txt from experiment save_dir. Returns (phase, epoch, val_loss)."""
    if not save_dir:
        return ""
    progress_file = os.path.join(save_dir, "progress.txt")
    try:
        with open(progress_file) as f:
            parts = f.read().strip().split(",")
            if len(parts) >= 2:
                phase = parts[0]  # training / eval_asv19 / eval_df21
                epoch = parts[1]  # e.g. 5/100
                return f"{phase} {epoch}"
    except FileNotFoundError:
        pass
    return "starting..."


def format_lr(lr):
    if lr == 1e-5:
        return "1e-5"
    elif lr == 5e-5:
        return "5e-5"
    elif lr == 1e-4:
        return "1e-4"
    return f"{lr:.0e}"


def status_style(status):
    styles = {
        "pending": "dim",
        "running": "bold yellow",
        "done": "bold green",
        "error": "bold red",
    }
    return styles.get(status, "")


def build_table(experiments):
    table = Table(title="D1 LoRA HP Search Dashboard", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", width=13)
    table.add_column("lr", width=8)
    table.add_column("r", width=4, justify="right")
    table.add_column("alpha", width=5, justify="right")
    table.add_column("Status", width=20)
    table.add_column("Val Loss", width=10, justify="right")
    table.add_column("ASV19 D1 EER", width=13, justify="right")
    table.add_column("DF21 D1 EER", width=13, justify="right")

    for exp in experiments:
        status = exp["status"]
        val_loss = f"{exp['best_val_loss']:.4f}" if exp["best_val_loss"] is not None else "-"
        asv = f"{exp['eer_asv19_d1']:.4f}%" if exp["eer_asv19_d1"] is not None else "-"
        df = f"{exp['eer_df21_d1']:.4f}%" if exp["eer_df21_d1"] is not None else "-"

        style = status_style(status)
        if status == "running":
            progress = read_progress(exp.get("save_dir"))
            status_display = f"[{style}]{progress}[/{style}]" if HAS_RICH else progress
        else:
            status_display = f"[{style}]{status}[/{style}]" if HAS_RICH else status
        table.add_row(
            str(exp["id"]),
            exp["model"],
            format_lr(exp["lr"]),
            str(exp["r"]),
            str(exp["alpha"]),
            status_display,
            val_loss, asv, df
        )

    return table


def build_summary(experiments):
    counts = {}
    for e in experiments:
        counts[e["status"]] = counts.get(e["status"], 0) + 1

    lines = []
    parts = []
    for s in ["pending", "running", "done", "error"]:
        if counts.get(s, 0) > 0:
            parts.append(f"{s}: {counts[s]}")
    lines.append(" | ".join(parts))

    done_exps = [e for e in experiments if e["status"] == "done"]
    if done_exps:
        best_asv = min(done_exps, key=lambda e: e["eer_asv19_d1"] if e["eer_asv19_d1"] is not None else 999)
        best_df = min(done_exps, key=lambda e: e["eer_df21_d1"] if e["eer_df21_d1"] is not None else 999)

        if best_asv["eer_asv19_d1"] is not None:
            lines.append(f"Best ASV19 D1: {best_asv['eer_asv19_d1']:.4f}% "
                         f"({best_asv['model']}, lr={format_lr(best_asv['lr'])}, r={best_asv['r']}, alpha={best_asv['alpha']})")
        if best_df["eer_df21_d1"] is not None:
            lines.append(f"Best DF21 D1:  {best_df['eer_df21_d1']:.4f}% "
                         f"({best_df['model']}, lr={format_lr(best_df['lr'])}, r={best_df['r']}, alpha={best_df['alpha']})")

    return "\n".join(lines)


def show_once():
    experiments = get_experiments()
    if not experiments:
        print("No experiments found. Run generate_experiments.py first.")
        return

    if HAS_RICH:
        console = Console()
        table = build_table(experiments)
        console.print(table)
        summary = build_summary(experiments)
        console.print(Panel(summary, title="Summary"))
    else:
        print(f"\n{'#':<3} {'Model':<13} {'lr':<8} {'r':>4} {'alpha':>5} {'Status':<9} {'ValLoss':>10} {'ASV19 D1':>13} {'DF21 D1':>13}")
        print("-" * 88)
        for e in experiments:
            val = f"{e['best_val_loss']:.4f}" if e['best_val_loss'] else "-"
            asv = f"{e['eer_asv19_d1']:.4f}%" if e['eer_asv19_d1'] else "-"
            df = f"{e['eer_df21_d1']:.4f}%" if e['eer_df21_d1'] else "-"
            print(f"{e['id']:<3} {e['model']:<13} {format_lr(e['lr']):<8} {e['r']:>4} {e['alpha']:>5} {e['status']:<9} {val:>10} {asv:>13} {df:>13}")
        print()
        print(build_summary(experiments))


def show_watch():
    if not HAS_RICH:
        print("Rich library required for --watch mode. Install: pip install rich")
        return

    console = Console()
    with Live(console=console, refresh_per_second=0.1) as live:
        while True:
            experiments = get_experiments()
            if not experiments:
                break

            table = build_table(experiments)
            summary = build_summary(experiments)

            from rich.layout import Layout
            from rich.text import Text

            output = Table.grid()
            output.add_row(table)
            output.add_row(Panel(summary, title="Summary"))
            output.add_row(Text(f"Last updated: {time.strftime('%H:%M:%S')} (Ctrl+C to exit)", style="dim"))

            live.update(output)

            counts = {}
            for e in experiments:
                counts[e["status"]] = counts.get(e["status"], 0) + 1
            if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
                break

            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HP Search Dashboard")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh every 10s")
    args = parser.parse_args()

    if args.watch:
        try:
            show_watch()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
    else:
        show_once()
