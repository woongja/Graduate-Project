"""
AutoResearch LoRA — 1-step 파이프라인 테스트

각 모델에 대해:
  1. pretrained 로드 → LoRA 적용 → 1 step 학습 → 정상 종료 확인
  2. eval 1 batch → score 출력 확인

Usage:
    python scripts/auto_lora/test.py --gpu MIG-xxx
    python scripts/auto_lora/test.py --gpu cuda:0 --model aasist  # 특정 모델만
"""

import argparse
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
}

PROTO_TRAIN = os.path.join(BASE_DIR, "protocols", "asv19_train.txt")
PROTO_DEV = os.path.join(BASE_DIR, "protocols", "asv19_dev.txt")
PROTO_EVAL = os.path.join(BASE_DIR, "protocols", "asv19_eval.txt")


def test_model(model_name, gpu):
    """Test 1-step train + eval for a model."""
    cfg = MODEL_CONFIGS[model_name]
    save_dir = os.path.join(BASE_DIR, "out", "auto_lora_test", model_name)
    os.makedirs(save_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = "4"

    print(f"\n{'─'*50}")
    print(f"  Testing: {model_name}")
    print(f"{'─'*50}")

    # ── Train 1 epoch (patience=1, max_epochs=1) ──
    print(f"  [1/2] Training (1 epoch)...", end=" ", flush=True)
    train_cmd = [
        "python", "main.py",
        "--config", cfg["config"],
        "--model_path", cfg["pretrained"],
        "--train_protocol", PROTO_TRAIN,
        "--dev_protocol", PROTO_DEV,
        "--domain", "1",
        "--lora",
        "--lora_r", "4",
        "--lora_alpha", "8",
        "--learning_rate", "1e-4",
        "--batch_size", "4",
        "--max_epochs", "1",
        "--patience", "1",
        "--save_dir", save_dir,
    ]

    log_file = os.path.join(save_dir, "test_train.log")
    with open(log_file, "w") as lf:
        result = subprocess.run(
            train_cmd, cwd=BASE_DIR,
            stdout=lf, stderr=lf,
            env=env, timeout=600
        )

    if result.returncode != 0:
        print("FAIL")
        print(f"    Log: {log_file}")
        with open(log_file) as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"    {line.rstrip()}")
        return False

    # Check checkpoint exists
    ckpt = os.path.join(save_dir, "D1_best.pth")
    if not os.path.exists(ckpt):
        print("FAIL (no checkpoint)")
        return False
    print("OK")

    # ── Eval ──
    print(f"  [2/2] Eval...", end=" ", flush=True)
    eval_output = os.path.join(save_dir, "test_scores.txt")
    eval_cmd = [
        "python", "main.py",
        "--config", cfg["config"],
        "--eval", "--nc_eval",
        "--model_path", ckpt,
        "--protocol_path", PROTO_EVAL,
        "--domain", "1",
        "--lora_r", "4",
        "--lora_alpha", "8",
        "--eval_output", eval_output,
        "--batch_size", "4",
    ]

    eval_log = os.path.join(save_dir, "test_eval.log")
    with open(eval_log, "w") as lf:
        result = subprocess.run(
            eval_cmd, cwd=BASE_DIR,
            stdout=lf, stderr=lf,
            env=env, timeout=600
        )

    if result.returncode != 0:
        print("FAIL")
        print(f"    Log: {eval_log}")
        with open(eval_log) as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"    {line.rstrip()}")
        return False

    # Check score file
    if os.path.exists(eval_output) and os.path.getsize(eval_output) > 0:
        print("OK")
    else:
        print("FAIL (empty scores)")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Test AutoResearch pipeline")
    parser.add_argument("--gpu", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        choices=["aasist", "conformertcm", "xlsr_sls", "xlsr_mamba"])
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIGS.keys())

    print(f"{'='*50}")
    print(f" AutoResearch Pipeline Test")
    print(f"  Models: {models}")
    print(f"  GPU: {args.gpu}")
    print(f"{'='*50}")

    results = {}
    for model in models:
        results[model] = test_model(model, args.gpu)

    # Summary
    print(f"\n{'='*50}")
    print(f" Summary")
    print(f"{'='*50}")
    all_pass = True
    for model, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {model}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  All tests passed!")
    else:
        print(f"\n  Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
