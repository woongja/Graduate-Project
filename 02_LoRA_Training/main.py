"""
LoRA Training - Train & Eval Script

Usage:
  # Eval
  python main.py --config config/aasist_baseline.yaml --eval \
      --model_path pretrained/aasist.pth --protocol_path protocols/original/protocol_asv19_eval.txt \
      --database_path /path/to/ASVspoof2019_eval --eval_output results/scores.txt

  # Train (domain finetuning)
  python main.py --config config/conformertcm_baseline.yaml \
      --model_path pretrained/conformertcm.pth \
      --train_protocol protocols/asv19_train.txt --dev_protocol protocols/asv19_dev.txt \
      --domain 1 --lora --save_dir out/conformertcm_D1
"""

import argparse
import logging
import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import numpy as np

logging.getLogger("numba").setLevel(logging.WARNING)
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
from tensorboardX import SummaryWriter

from core_scripts.startup_config import set_random_seed


# ========== Training ==========

def train_epoch(train_loader, model, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    num_batches = 0

    pbar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch} [Train]")
    for batch_x, batch_target in pbar:
        batch_x = batch_x.to(device)
        batch_target = batch_target.to(device).long()

        out = model(batch_x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion(logits, batch_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        pred = logits.argmax(dim=1)
        correct += (pred == batch_target).sum().item()
        total += batch_target.size(0)

        pbar.set_postfix({
            'loss': f'{running_loss / num_batches:.4f}',
            'acc': f'{correct / total * 100:.2f}%'
        })

    avg_loss = running_loss / num_batches
    acc = correct / total * 100
    return avg_loss, acc


def eval_epoch(dev_loader, model, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    num_batches = 0

    pbar = tqdm(dev_loader, ncols=120, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch_x, batch_target in pbar:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device).long()

            out = model(batch_x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, batch_target)

            val_loss += loss.item()
            num_batches += 1

            pred = logits.argmax(dim=1)
            correct += (pred == batch_target).sum().item()
            total += batch_target.size(0)

            pbar.set_postfix({
                'loss': f'{val_loss / num_batches:.4f}',
                'acc': f'{correct / total * 100:.2f}%'
            })

    avg_loss = val_loss / num_batches
    acc = correct / total * 100
    return avg_loss, acc


# ========== Evaluation (inference + scores) ==========

def eval_model(args, config, device):
    """Evaluation mode: load model, run inference, save scores."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config.get("training", {})

    for key, value in model_cfg.items():
        if key != "name":
            setattr(args, key, value)
    for key, value in train_cfg.items():
        setattr(args, key, value)

    # Dataset
    data_module = importlib.import_module("datautils." + data_cfg["name"])

    if args.domain is not None or args.nc_eval:
        # NC protocol: absolute_path label aug_type
        _, file_eval_list = data_module.genDomain_list(args.protocol_path, domain=args.domain)
        file_eval = file_eval_list
        base_dir = ""
        if args.domain is not None:
            print(f"[INFO] Domain eval: D{args.domain} | Samples: {len(file_eval)}")
        else:
            print(f"[INFO] NC eval samples: {len(file_eval)}")
    else:
        file_eval = data_module.genSpoof_list(args.protocol_path, is_eval=True)
        base_dir = args.database_path or ""
        print(f"[INFO] Eval samples: {len(file_eval)}")

    eval_set = data_module.Dataset_eval(
        list_IDs=file_eval, base_dir=base_dir, cut=data_cfg.get("cut", 64600)
    )
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    modelClass = importlib.import_module("model." + model_cfg["name"]).Model
    model = modelClass(args, device).to(device)

    if args.model_path:
        if args.model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(args.model_path)
        else:
            ckpt = torch.load(args.model_path, map_location="cpu")
        is_lora_ckpt = any(k.startswith("base_model.") for k in ckpt.keys())

        if is_lora_ckpt:
            # LoRA checkpoint: apply LoRA first, then load
            from model.lora_wrapper import apply_lora
            lora_cfg = config.get("lora", {})
            lora_r = args.lora_r if args.lora_r else lora_cfg.get("r", 8)
            lora_alpha_val = args.lora_alpha if args.lora_alpha else lora_cfg.get("alpha", 16)
            model = apply_lora(
                model,
                r=lora_r,
                lora_alpha=lora_alpha_val,
                target_modules=lora_cfg.get("target_modules", None),
                lora_dropout=lora_cfg.get("dropout", 0.0),
            )
            model.load_state_dict(ckpt)
            print(f"[INFO] Loaded LoRA checkpoint: {args.model_path}")
        else:
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt, strict=False)
            print(f"[INFO] Loaded checkpoint: {args.model_path}")

    model.eval()

    # Inference
    os.makedirs(os.path.dirname(args.eval_output) if os.path.dirname(args.eval_output) else ".", exist_ok=True)
    with torch.no_grad(), open(args.eval_output, "w") as fh:
        for batch_x, utt_id in tqdm(eval_loader, desc="Evaluation", leave=False):
            batch_x = batch_x.to(device)
            out = model(batch_x)
            logits = out[0] if isinstance(out, tuple) else out
            scores = logits.cpu().numpy().tolist()
            for f, s in zip(utt_id, scores):
                fh.write(f"{f} {s[0]} {s[1]}\n")

    print(f"[INFO] Scores saved to: {args.eval_output}")


# ========== Main ==========

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    set_random_seed(args.seed)

    # ========== EVAL MODE ==========
    if args.eval:
        eval_model(args, config, device)
        sys.exit(0)

    # ========== TRAIN MODE ==========
    data_cfg = config["data"]
    model_cfg = config["model"]

    # Inject model config into args
    for key, value in model_cfg.items():
        if key != "name":
            setattr(args, key, value)

    # ── Dataset ──
    data_module = importlib.import_module("datautils." + data_cfg["name"])

    train_protocol = args.train_protocol or data_cfg.get("train_protocol")
    dev_protocol = args.dev_protocol or data_cfg.get("dev_protocol")

    if not train_protocol or not dev_protocol:
        print("[ERROR] --train_protocol and --dev_protocol are required for training")
        sys.exit(1)

    d_label_trn, file_train = data_module.genDomain_list(train_protocol, domain=args.domain)
    d_label_dev, file_dev = data_module.genDomain_list(dev_protocol, domain=args.domain)

    domain_str = f"D{args.domain}" if args.domain is not None else "all"
    print(f"[INFO] Domain: {domain_str}")
    print(f"[INFO] Train: {len(file_train)} samples, Dev: {len(file_dev)} samples")

    # base_dir="" because protocol has absolute paths
    train_set = data_module.Dataset_train(
        args=args, list_IDs=file_train, labels=d_label_trn,
        base_dir="", algo=0
    )
    dev_set = data_module.Dataset_train(
        args=args, list_IDs=file_dev, labels=d_label_dev,
        base_dir="", algo=0
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── Model ──
    modelClass = importlib.import_module("model." + model_cfg["name"]).Model
    model = modelClass(args, device).to(device)

    # Load pretrained checkpoint
    if args.model_path:
        if args.model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(args.model_path)
        else:
            ckpt = torch.load(args.model_path, map_location="cpu")
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=False)
        print(f"[INFO] Loaded pretrained: {args.model_path}")

    # Apply LoRA
    if args.lora:
        from model.lora_wrapper import apply_lora
        lora_cfg = config.get("lora", {})
        lora_r = args.lora_r if args.lora_r else lora_cfg.get("r", 8)
        lora_alpha = args.lora_alpha if args.lora_alpha else lora_cfg.get("alpha", 16)
        model = apply_lora(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_cfg.get("target_modules", None),
            lora_dropout=lora_cfg.get("dropout", 0.0),
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total params: {total_params:,} | Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # ── Optimizer & Loss ──
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"[INFO] Optimizer: AdamW lr={args.learning_rate:.2e}")

    class_weights = config.get("training", {}).get("class_weights", [0.9, 0.1])
    weight = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    print(f"[INFO] CE weights: {class_weights}")

    # ── Save dir ──
    model_name = model_cfg["name"]
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = f"out/{model_name}_D{args.domain}" if args.domain is not None else f"out/{model_name}_finetune"
    os.makedirs(save_dir, exist_ok=True)
    if args.domain is not None:
        save_filename = f"D{args.domain}_best.pth"
    else:
        save_filename = "best_model.pth"
    model_save_path = os.path.join(save_dir, save_filename)

    log_dir = args.log_dir if args.log_dir else os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    log_file = os.path.join(log_dir, "training.log")
    with open(log_file, "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    # ── Training Loop ──
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n{'=' * 80}")
    print(f"Starting Training: {model_name} | {domain_str}")
    print(f"  LoRA: {'Yes' if args.lora else 'No'}")
    print(f"  Model save: {model_save_path}")
    print(f"  Epochs: {args.max_epochs} | Patience: {args.patience}")
    print(f"{'=' * 80}\n")

    start_time_total = time.time()

    # Initial progress file (for dashboard monitoring)
    progress_file = os.path.join(save_dir, "progress.txt")
    with open(progress_file, "w") as f:
        f.write(f"training,0/{args.max_epochs},-")

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()

        tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, criterion, device, epoch)
        val_loss, val_acc = eval_epoch(dev_loader, model, criterion, device, epoch)

        epoch_time = time.time() - epoch_start

        # TensorBoard
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Console
        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"{'─' * 80}")

        # CSV Log
        with open(log_file, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.2f},{val_loss:.4f},{val_acc:.2f}\n")

        # Progress file (for external monitoring)
        with open(progress_file, "w") as f:
            f.write(f"training,{epoch}/{args.max_epochs},{val_loss:.4f}")

        # Checkpoint (best val loss)
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  Val loss improved by {improvement:.4f}! Model saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    writer.close()

    total_time = time.time() - start_time_total
    print(f"\n{'=' * 80}")
    print(f"Training Complete! Total: {total_time / 60:.2f} min | Best val loss: {best_val_loss:.4f}")
    print(f"Model: {model_save_path} | Logs: {log_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Training & Evaluation")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=24)

    # Training args
    parser.add_argument("--train_protocol", type=str, default=None)
    parser.add_argument("--dev_protocol", type=str, default=None)
    parser.add_argument("--domain", type=int, default=None, help="Domain ID (0-7)")
    parser.add_argument("--lora", action="store_true", help="Apply LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank (overrides config)")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (overrides config)")
    parser.add_argument("--model_path", type=str, default=None, help="Pretrained checkpoint path")
    parser.add_argument("--save_dir", type=str, default=None, help="Model save directory")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)

    # Eval args
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")
    parser.add_argument("--nc_eval", action="store_true", help="NC protocol format (abs_path label aug_type)")
    parser.add_argument("--database_path", type=str, default=None)
    parser.add_argument("--protocol_path", type=str, default=None)
    parser.add_argument("--eval_output", type=str, default="results/eval_scores.txt")

    # RawBoost args (used by Dataset_train)
    parser.add_argument('--algo', type=int, default=0)
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)

    args = parser.parse_args()
    main(args)
