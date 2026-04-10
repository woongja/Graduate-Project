"""
AutoResearch LoRA Dashboard

Usage:
    python scripts/auto_lora/dashboard.py
    python scripts/auto_lora/dashboard.py --watch
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "out", "auto_lora")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

MODELS = ["aasist", "conformertcm", "xlsr_sls"]


def load_results(model):
    """Load results.tsv for a model."""
    tsv = os.path.join(RESULTS_DIR, model, "results.tsv")
    if not os.path.exists(tsv):
        return []
    rows = []
    with open(tsv) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 11:
                rows.append({
                    "trial": parts[0],
                    "model": parts[1],
                    "domain": parts[2],
                    "lr": parts[3],
                    "r": parts[4],
                    "alpha": parts[5],
                    "dropout": parts[6],
                    "batch_size": parts[7],
                    "val_loss": parts[8],
                    "asv19_eer": parts[9],
                    "df21_eer": parts[10],
                })
    return rows


def load_running_trials(model):
    """Find currently running trials by checking progress.txt files."""
    model_dir = os.path.join(RESULTS_DIR, model)
    if not os.path.isdir(model_dir):
        return []
    running = []
    for d in sorted(os.listdir(model_dir)):
        if not d.startswith("trial_"):
            continue
        progress_file = os.path.join(model_dir, d, "progress.txt")
        if not os.path.exists(progress_file):
            continue
        try:
            with open(progress_file) as f:
                content = f.read().strip()
            # format: training,{epoch}/{max_epochs},{val_loss}
            parts = content.split(",")
            if len(parts) >= 3 and parts[0] == "training":
                epoch_info = parts[1]  # e.g. "3/100"
                val_loss = parts[2]
                trial_num = d.replace("trial_", "")
                running.append({
                    "trial": trial_num,
                    "epoch": epoch_info,
                    "val_loss": val_loss,
                })
        except Exception:
            pass
    return running


def load_optuna_info(model, domain=1):
    """Load Optuna study info: best trial + running trial params."""
    db_path = os.path.join(RESULTS_DIR, model, f"optuna_D{domain}.db")
    if not os.path.exists(db_path):
        return None, {}
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.load_study(
            study_name=f"{model}_D{domain}",
            storage=f"sqlite:///{db_path}",
        )
        best_info = None
        try:
            bt = study.best_trial
        except ValueError:
            bt = None
        if bt:
            best_info = {
                "trial": study.best_trial.number,
                "eer": study.best_value,
                "params": study.best_params,
                "n_complete": len([t for t in study.trials if t.state.name == "COMPLETE"]),
                "n_running": len([t for t in study.trials if t.state.name == "RUNNING"]),
                "n_fail": len([t for t in study.trials if t.state.name == "FAIL"]),
            }
        # Get params for running trials
        running_params = {}
        for t in study.trials:
            if t.state.name == "RUNNING":
                p = t.params
                alpha = p.get("r", 0) * p.get("alpha_mult", 1)
                running_params[str(t.number)] = {
                    "lr": f"{p.get('lr', 0):.2e}",
                    "r": str(p.get("r", "")),
                    "alpha": str(alpha),
                    "dropout": f"{p.get('dropout', 0):.2f}",
                    "batch_size": str(p.get("batch_size", "")),
                }
        return best_info, running_params
    except Exception:
        pass
    return None, {}


def show_dashboard():
    if HAS_RICH:
        console = Console()
    else:
        console = None

    for model in MODELS:
        rows = load_results(model)
        best_info, running_params = load_optuna_info(model)
        running = load_running_trials(model)

        # Filter out completed trials from running list
        completed_ids = {r["trial"] for r in rows}
        running = [r for r in running if r["trial"] not in completed_ids]

        if HAS_RICH:
            # Results table
            table = Table(title=f"AutoResearch — {model}", show_lines=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("lr", width=10)
            table.add_column("r", width=4, justify="right")
            table.add_column("alpha", width=5, justify="right")
            table.add_column("drop", width=5, justify="right")
            table.add_column("bs", width=4, justify="right")
            table.add_column("Val Loss", width=10, justify="right")
            table.add_column("ASV19 EER", width=10, justify="right")
            table.add_column("DF21 EER", width=10, justify="right")
            table.add_column("Progress", width=20, justify="left")

            for r in rows:
                asv19_val = r["asv19_eer"]
                df21_val = r["df21_eer"]
                asv19_style = ""
                if asv19_val != "-":
                    try:
                        if best_info and float(asv19_val) <= best_info["eer"] + 0.01:
                            asv19_style = "bold green"
                    except ValueError:
                        pass
                asv19_display = f"[{asv19_style}]{asv19_val}%[/{asv19_style}]" if asv19_style else (f"{asv19_val}%" if asv19_val != "-" else "-")
                df21_display = f"{df21_val}%" if df21_val != "-" else "-"
                table.add_row(
                    r["trial"], r["lr"], r["r"], r["alpha"],
                    r["dropout"], r["batch_size"], r["val_loss"],
                    asv19_display, df21_display, "[green]Done[/green]",
                )

            # Show running trials with HP from Optuna
            for rt in running:
                epoch_cur, epoch_max = rt["epoch"].split("/")
                pct = int(float(epoch_cur) / float(epoch_max) * 100) if float(epoch_max) > 0 else 0
                bar_len = 10
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                progress_str = f"[yellow]{bar} {rt['epoch']}[/yellow]"

                hp = running_params.get(rt["trial"], {})
                table.add_row(
                    rt["trial"],
                    hp.get("lr", ""),
                    hp.get("r", ""),
                    hp.get("alpha", ""),
                    hp.get("dropout", ""),
                    hp.get("batch_size", ""),
                    rt["val_loss"],
                    "-", "-",
                    progress_str,
                )

            console.print(table)

            # Summary
            lines = []
            if best_info:
                lines.append(f"Completed: {best_info['n_complete']} | Running: {best_info['n_running']} | Failed: {best_info['n_fail']}")
                lines.append(f"Best ASV19 EER: {best_info['eer']:.4f}% (trial #{best_info['trial']})")
                params = best_info["params"]
                lines.append(f"  lr={params.get('lr', '?'):.2e} r={params.get('r', '?')} "
                             f"alpha_mult={params.get('alpha_mult', '?')} dropout={params.get('dropout', '?')}")
            else:
                lines.append(f"Completed: {len(rows)} | No best trial yet")

            if running:
                lines.append(f"Training: {len(running)} trial(s) in progress")

            console.print(Panel("\n".join(lines), title=f"{model} Summary"))
        else:
            print(f"\n=== {model} ===")
            print(f"{'#':<4} {'lr':<10} {'r':>4} {'alpha':>5} {'drop':>5} {'bs':>4} {'val_loss':>10} {'ASV19':>10} {'DF21':>10} {'Progress':>15}")
            print("-" * 90)
            for r in rows:
                asv19 = f"{r['asv19_eer']}%" if r['asv19_eer'] != "-" else "-"
                df21 = f"{r['df21_eer']}%" if r['df21_eer'] != "-" else "-"
                print(f"{r['trial']:<4} {r['lr']:<10} {r['r']:>4} {r['alpha']:>5} "
                      f"{r['dropout']:>5} {r['batch_size']:>4} {r['val_loss']:>10} {asv19:>10} {df21:>10} {'Done':>15}")
            for rt in running:
                hp = running_params.get(rt["trial"], {})
                epoch_cur, epoch_max = rt["epoch"].split("/")
                pct = int(float(epoch_cur) / float(epoch_max) * 100) if float(epoch_max) > 0 else 0
                print(f"{rt['trial']:<4} {hp.get('lr',''):<10} {hp.get('r',''):>4} {hp.get('alpha',''):>5} "
                      f"{hp.get('dropout',''):>5} {hp.get('batch_size',''):>4} {rt['val_loss']:>10} {'-':>10} {'-':>10} {rt['epoch']+f' ({pct}%)':>15}")
            if best_info:
                print(f"\nBest ASV19 EER: {best_info['eer']:.4f}% (trial #{best_info['trial']})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system("clear")
                show_dashboard()
                print(f"\nLast updated: {time.strftime('%H:%M:%S')} — Ctrl+C to exit")
                time.sleep(15)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
    else:
        show_dashboard()
