"""
AutoResearch LoRA — Optuna 기반 자동 HP 탐색

각 trial:
  1. Optuna가 HP 샘플링 (lr, r, alpha, dropout, batch_size)
  2. main.py로 학습 (특정 도메인, patience=5)
  3. 해당 도메인 ASV19 eval → EER 계산
  4. Optuna에 EER 보고 (minimize)
  5. results.tsv에 누적

Usage:
    python scripts/auto_lora/runner.py --model aasist --gpu MIG-xxx --domain 1 --n_trials 50
"""

import argparse
import csv
import os
import subprocess
import sys
import yaml
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import optuna

# 새 도메인 매핑
DOMAIN_TO_AUGTYPES = {
    0: ["clean"],
    1: ["background_noise"],
    2: ["background_music"],
    3: ["auto_tune"],
    4: ["high_pass_filter", "low_pass_filter"],
    5: ["echo"],
    6: ["pitch_shift"],
    7: ["time_stretch"],
    8: ["gaussian_noise"],
    9: ["reverberation"],
}

MODEL_CONFIGS = {
    "aasist": {
        "config": "config/aasist_baseline.yaml",
        "pretrained": "pretrained/aasist.pth",
    },
    "conformertcm": {
        "config": "config/conformertcm_baseline.yaml",
        "pretrained": "pretrained/conformertcm.pth",
    },
    "xlsr_sls": {
        "config": "config/xlsr_sls_baseline.yaml",
        "pretrained": "pretrained/XLSR-SLS.pth",
    },
    "xlsr_mamba": {
        "config": "config/xlsr_mamba_baseline.yaml",
        "pretrained": "pretrained/XLSR-Mamba-DF/model.safetensors",
    },
}

RESULTS_DIR = os.path.join(BASE_DIR, "out", "auto_lora")
PROTO_ASV19 = os.path.join(BASE_DIR, "protocols", "asv19_eval.txt")
PROTO_DF21 = os.path.join(BASE_DIR, "protocols", "df21_eval.txt")
PROTO_TRAIN = os.path.join(BASE_DIR, "protocols", "asv19_train.txt")
PROTO_DEV = os.path.join(BASE_DIR, "protocols", "asv19_dev.txt")


def run_trial(model_name, domain, lr, r, alpha, dropout, batch_size, gpu, trial_id):
    """Run a single training + eval trial."""
    model_cfg = MODEL_CONFIGS[model_name]
    save_dir = os.path.join(RESULTS_DIR, model_name, f"trial_{trial_id}")
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "train.log")

    # ── Train ──
    train_cmd = [
        "python", "main.py",
        "--config", model_cfg["config"],
        "--model_path", model_cfg["pretrained"],
        "--train_protocol", PROTO_TRAIN,
        "--dev_protocol", PROTO_DEV,
        "--domain", str(domain),
        "--lora",
        "--lora_r", str(r),
        "--lora_alpha", str(alpha),
        "--learning_rate", str(lr),
        "--batch_size", str(batch_size),
        "--max_epochs", "100",
        "--patience", "5",
        "--save_dir", save_dir,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = "4"

    with open(log_file, "w") as lf:
        result = subprocess.run(
            train_cmd, cwd=BASE_DIR,
            stdout=lf, stderr=lf,
            env=env, timeout=7200  # 2hr max
        )

    if result.returncode != 0:
        print(f"  [FAIL] Training failed (trial {trial_id})")
        return None, None

    # ── Extract best val loss ──
    best_val_loss = 999.0
    train_log = os.path.join(save_dir, "logs", "training.log")
    if os.path.exists(train_log):
        import csv as csv_mod
        with open(train_log) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                vl = float(row["Val_Loss"])
                if vl < best_val_loss:
                    best_val_loss = vl

    # ── Eval on domain ──
    ckpt = os.path.join(save_dir, f"D{domain}_best.pth")
    if not os.path.exists(ckpt):
        print(f"  [FAIL] Checkpoint not found: {ckpt}")
        return best_val_loss, None, None

    def eval_dataset(proto_path, proto_name):
        """Run eval + compute EER for a dataset."""
        eval_output = os.path.join(save_dir, f"{proto_name}_d{domain}_scores.txt")
        eval_cmd = [
            "python", "main.py",
            "--config", model_cfg["config"],
            "--eval", "--nc_eval",
            "--model_path", ckpt,
            "--protocol_path", proto_path,
            "--domain", str(domain),
            "--lora_r", str(r),
            "--lora_alpha", str(alpha),
            "--eval_output", eval_output,
            "--batch_size", "64",
        ]

        eval_log = os.path.join(save_dir, f"{proto_name}_eval.log")
        with open(eval_log, "w") as lf:
            res = subprocess.run(
                eval_cmd, cwd=BASE_DIR,
                stdout=lf, stderr=lf,
                env=env, timeout=3600
            )

        if res.returncode != 0:
            print(f"  [FAIL] {proto_name} eval failed (trial {trial_id})")
            return None

        try:
            import pandas as pd
            from evaluate_metrics import compute_eer

            proto = pd.read_csv(proto_path, sep=" ", header=None, names=["filepath", "label", "aug_type"])
            allowed = DOMAIN_TO_AUGTYPES[domain]
            proto = proto[proto["aug_type"].isin(allowed)]

            score = pd.read_csv(eval_output, sep=" ", header=None, names=["utt", "spoof", "bonafide"])
            merged = pd.merge(proto, score, left_on="filepath", right_on="utt")

            bf = merged[merged["label"] == "bonafide"]["bonafide"].values
            sp = merged[merged["label"] == "spoof"]["bonafide"].values

            if len(bf) > 0 and len(sp) > 0:
                eer, _ = compute_eer(bf, sp)
                return eer * 100
        except Exception as e:
            print(f"  [FAIL] {proto_name} EER computation failed: {e}")

        return None

    asv19_eer = eval_dataset(PROTO_ASV19, "asv19")
    df21_eer = eval_dataset(PROTO_DF21, "df21")

    return best_val_loss, asv19_eer, df21_eer


def create_objective(model_name, domain, gpu, results_file):
    """Create Optuna objective function."""
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        r = trial.suggest_categorical("r", [4, 8, 16, 32])
        alpha_mult = trial.suggest_categorical("alpha_mult", [1, 2, 4])
        alpha = r * alpha_mult
        dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
        batch_size = trial.suggest_categorical("batch_size", [16, 24, 32])

        trial_id = trial.number
        print(f"\n{'='*60}")
        print(f"Trial {trial_id}: {model_name} D{domain}")
        print(f"  lr={lr:.2e} r={r} alpha={alpha} dropout={dropout} bs={batch_size}")
        print(f"{'='*60}")

        val_loss, asv19_eer, df21_eer = run_trial(
            model_name, domain, lr, r, alpha, dropout, batch_size, gpu, trial_id
        )

        # Record results
        with open(results_file, "a", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow([
                trial_id, model_name, domain, lr, r, alpha, dropout, batch_size,
                f"{val_loss:.4f}" if val_loss and val_loss < 999 else "-",
                f"{asv19_eer:.4f}" if asv19_eer is not None else "-",
                f"{df21_eer:.4f}" if df21_eer is not None else "-",
                datetime.now().isoformat(),
            ])

        if asv19_eer is None:
            return float("inf")  # failed trial

        print(f"  → ASV19 EER: {asv19_eer:.4f}% | DF21 EER: {df21_eer:.4f}%" if df21_eer is not None
              else f"  → ASV19 EER: {asv19_eer:.4f}% | DF21 EER: -")
        return asv19_eer

    return objective


def main():
    parser = argparse.ArgumentParser(description="AutoResearch LoRA HP Search")
    parser.add_argument("--model", type=str, required=True,
                        choices=["aasist", "conformertcm", "xlsr_sls"])
    parser.add_argument("--gpu", type=str, required=True)
    parser.add_argument("--domain", type=int, default=1, help="Domain to optimize (0-9)")
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()

    # Setup results file
    results_dir = os.path.join(RESULTS_DIR, args.model)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "results.tsv")

    # Write header if new
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["trial", "model", "domain", "lr", "r", "alpha",
                         "dropout", "batch_size", "val_loss", "asv19_eer", "df21_eer", "timestamp"])

    print(f"{'='*60}")
    print(f" AutoResearch LoRA — {args.model}")
    print(f"  Domain: D{args.domain}")
    print(f"  GPU: {args.gpu}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Results: {results_file}")
    print(f"{'='*60}")

    # Optuna study
    db_path = os.path.join(results_dir, f"optuna_D{args.domain}.db")
    study = optuna.create_study(
        study_name=f"{args.model}_D{args.domain}",
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = create_objective(args.model, args.domain, args.gpu, results_file)
    study.optimize(objective, n_trials=args.n_trials)

    # Print best
    print(f"\n{'='*60}")
    print(f" Best Trial: {study.best_trial.number}")
    print(f"  EER: {study.best_value:.4f}%")
    print(f"  Params: {study.best_params}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
