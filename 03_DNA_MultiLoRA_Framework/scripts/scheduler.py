"""
Pipeline Eval Scheduler - GPU memory check, 1 launch per cycle.

Usage:
    python scripts/scheduler.py --gpus MIG-aaa,MIG-bbb
"""

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "results", "eval_pipeline.db")
RUN_SCRIPT = os.path.join(BASE_DIR, "scripts", "run_single.sh")

POLL_INTERVAL = 30
LAUNCH_COOLDOWN = 60
MIN_FREE_MB = 13312  # 13GB
MAX_PER_GPU = 1      # MIG당 최대 동시 실험 수


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_mig_free_memory_map():
    try:
        out_l = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
        mig_list = []
        gpu_idx = None
        for line in out_l.strip().split("\n"):
            gm = re.match(r"GPU (\d+):", line)
            if gm:
                gpu_idx = int(gm.group(1))
            mm = re.search(r"MIG.*Device\s+(\d+).*UUID:\s+(MIG-\S+)\)", line)
            if mm and gpu_idx is not None:
                mig_list.append((gpu_idx, int(mm.group(1)), mm.group(2)))

        out_q = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=60).stdout
        mig_blocks = re.findall(r"MIG Device\n(.*?)(?=MIG Device|Accounting Mode)", out_q, re.DOTALL)

        result = {}
        for i, block in enumerate(mig_blocks):
            free = re.search(r"FB Memory Usage.*?Free\s+:\s+(\d+)\s+MiB", block, re.DOTALL)
            if free and i < len(mig_list):
                result[mig_list[i][2]] = int(free.group(1))
        return result
    except Exception as e:
        print(f"[WARN] Failed to get MIG memory: {e}")
        return {}


def check_running(conn):
    rows = conn.execute("SELECT id, pid FROM experiments WHERE status='running'").fetchall()
    for row in rows:
        pid = row["pid"]
        if pid:
            try:
                os.kill(pid, 0)
            except OSError:
                print(f"[WARN] Experiment #{row['id']} (pid={pid}) died → error")
                conn.execute(
                    "UPDATE experiments SET status='error', end_time=? WHERE id=?",
                    (datetime.now().isoformat(), row["id"])
                )
                conn.commit()


def get_pending(conn):
    row = conn.execute(
        "SELECT id FROM experiments WHERE status='pending' ORDER BY id LIMIT 1"
    ).fetchone()
    return row["id"] if row else None


def get_counts(conn):
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM experiments GROUP BY status"
    ).fetchall()
    return {r["status"]: r["cnt"] for r in rows}


def launch(exp_id, gpu):
    log_dir = os.path.join(BASE_DIR, "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"exp_{exp_id}.log")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            ["bash", RUN_SCRIPT, str(exp_id), gpu],
            stdout=lf, stderr=lf,
            cwd=BASE_DIR,
            preexec_fn=os.setsid
        )

    print(f"  [LAUNCH] #{exp_id} on {gpu} (pid={proc.pid})")
    return proc.pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, required=True)
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",")]

    print(f"============================================================")
    print(f" Pipeline Eval Scheduler")
    print(f"  GPUs: {len(gpus)}")
    for g in gpus:
        print(f"    - {g}")
    print(f"  Min free: {MIN_FREE_MB}MB | Poll: {POLL_INTERVAL}s | Cooldown: {LAUNCH_COOLDOWN}s")
    print(f"  DB: {DB_PATH}")
    print(f"============================================================")
    print()

    if not os.path.exists(DB_PATH):
        print("[ERROR] DB not found. Run scripts/generate.py first.")
        sys.exit(1)

    try:
        while True:
            conn = get_db()
            check_running(conn)

            counts = get_counts(conn)
            pending = counts.get("pending", 0)
            running = counts.get("running", 0)
            done = counts.get("done", 0)
            error = counts.get("error", 0)

            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] pending={pending} running={running} done={done} error={error}")

            if pending == 0 and running == 0:
                print("All evaluations complete!")
                conn.close()
                break

            launched = False
            if pending > 0:
                mig_map = get_mig_free_memory_map()

                for gpu in gpus:
                    # Check max per GPU
                    gpu_running = conn.execute(
                        "SELECT COUNT(*) FROM experiments WHERE status='running' AND gpu=?", (gpu,)
                    ).fetchone()[0]
                    if gpu_running >= MAX_PER_GPU:
                        continue

                    free_mb = mig_map.get(gpu, 0)
                    if free_mb >= MIN_FREE_MB:
                        exp_id = get_pending(conn)
                        if exp_id:
                            pid = launch(exp_id, gpu)
                            conn.execute(
                                "UPDATE experiments SET status='running', pid=?, gpu=?, start_time=? WHERE id=?",
                                (pid, gpu, datetime.now().isoformat(), exp_id)
                            )
                            conn.commit()
                            launched = True
                            break
                    else:
                        if free_mb > 0:
                            print(f"  {gpu}: {free_mb}MB free (need {MIN_FREE_MB}MB)")

                if not launched:
                    print("  No GPU with enough memory, waiting...")

            conn.close()
            if launched:
                print(f"  Waiting {LAUNCH_COOLDOWN}s for model to load...")
                time.sleep(LAUNCH_COOLDOWN)
            else:
                time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scheduler stopped.")


if __name__ == "__main__":
    main()
