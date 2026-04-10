"""
DNA-MultiLoRA Evaluation Pipeline

Stage 1: waveform → NC predict → domain ID per sample
Stage 2: domain별 그룹핑 → ADD + LoRA 추론 → EER 계산

Usage:
    python eval.py --nc cnn8rnn_3ff_crossmodal --add aasist --dataset asv19
    python eval.py --nc ssast_tiny --add conformertcm --dataset df21
    python eval.py --nc htsat --add aasist --dataset itw --gpu MIG-xxx
"""

import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
logging.getLogger("numba").setLevel(logging.WARNING)

import yaml
import torch
import numpy as np
import pandas as pd
import librosa
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_BASE = os.path.abspath(os.path.join(BASE_DIR, "..", "02_LoRA_Training"))

sys.path.insert(0, BASE_DIR)
sys.path.insert(1, LORA_BASE)
from evaluate_metrics import compute_eer

DOMAIN_NAMES = {
    0: "clean", 1: "bg_noise+music", 2: "auto_tune", 3: "bandpass",
    4: "echo", 5: "pitch+stretch", 6: "gaussian", 7: "reverberation",
}


# ============================================================
# Unified waveform dataset (16kHz)
# ============================================================

class WaveformDataset(Dataset):
    """Load raw waveform at 16kHz. Supports NC and original protocol formats."""
    def __init__(self, protocol_path, proto_type="nc", base_dir="", cut=64600):
        self.cut = cut
        self.items = []
        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                if proto_type == "nc" and len(parts) >= 3:
                    self.items.append({
                        "filepath": parts[0],
                        "label": parts[1],  # spoof/bonafide
                    })
                elif proto_type == "original" and len(parts) >= 3:
                    filepath = os.path.join(base_dir, parts[0]) if base_dir else parts[0]
                    self.items.append({
                        "filepath": filepath,
                        "label": parts[2],
                    })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        x, _ = librosa.load(item["filepath"], sr=16000)
        if len(x) < self.cut:
            x = np.tile(x, (self.cut // len(x)) + 1)[:self.cut]
        else:
            x = x[:self.cut]
        return torch.FloatTensor(x), item["filepath"], item["label"]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DNA-MultiLoRA Evaluation")
    parser.add_argument("--config", type=str, default="config/pipeline.yaml")
    parser.add_argument("--nc", type=str, required=True)
    parser.add_argument("--add", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="cuda:0")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.gpu.startswith("MIG-") or args.gpu.startswith("GPU-"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = "cuda:0"
    else:
        device = args.gpu

    ds_cfg = config["eval"][args.dataset]
    protocol_path = os.path.join(BASE_DIR, ds_cfg["protocol"])
    proto_type = ds_cfg.get("type", "nc")
    base_dir = ds_cfg.get("base_dir", "")

    # ── Load waveform dataset ──
    dataset = WaveformDataset(protocol_path, proto_type=proto_type, base_dir=base_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"\n{'='*60}")
    print(f" DNA-MultiLoRA Evaluation")
    print(f"  NC: {args.nc} | ADD: {args.add} | Dataset: {args.dataset}")
    print(f"  Samples: {len(dataset)} | GPU: {args.gpu}")
    print(f"{'='*60}")

    # ── Stage 1: NC Inference ──
    print("\n[Stage 1] Noise Classification...")
    nc_config = config["noise_classifiers"][args.nc]

    # Map nc name to config name for yaml lookup
    nc_config_name_map = {
        "cnn8rnn_3ff_crossmodal": "cnn8rnn_3ff_crossmodal",
        "cnn8rnn_3ff_base": "cnn8rnn_3ff_base",
        "cnnlstm": "cnnlstm",
        "ssast_tiny": "ssast_tiny_patch_400",
        "htsat": "htsat",
    }

    from model.noise_classifier import NoiseClassifierWrapper
    nc_wrapper = NoiseClassifierWrapper(
        module_name=nc_config["module"],
        checkpoint_path=os.path.join(BASE_DIR, nc_config["checkpoint"]),
        nc_config_name=nc_config_name_map.get(args.nc, args.nc),
        device=device,
    )

    # Run NC on all samples
    file_domains = {}
    all_filepaths = []
    all_labels = []

    for audio, filepaths, labels in tqdm(dataloader, desc="NC Predict", leave=False):
        domain_ids, class_ids = nc_wrapper.predict(audio)
        for i in range(len(filepaths)):
            file_domains[filepaths[i]] = domain_ids[i].item()
            all_filepaths.append(filepaths[i])
            all_labels.append(labels[i])

    # Domain distribution
    domain_counts = defaultdict(int)
    for d in file_domains.values():
        domain_counts[d] += 1
    print(f"  Total: {len(file_domains)} samples")
    for d in sorted(domain_counts):
        print(f"  D{d} ({DOMAIN_NAMES.get(d, '?')}): {domain_counts[d]}")

    # ── Stage 2: ADD Inference ──
    # Keep NC in memory until ADD is loaded to prevent scheduler from
    # assigning another experiment to this GPU during the gap
    print("\n[Stage 2] ADD Inference...")
    add_config = config["add_models"][args.add]

    from model.add_model import ADDModelWrapper
    add_wrapper = ADDModelWrapper(
        model_module=add_config["model_module"],
        pretrained_path=os.path.join(BASE_DIR, add_config["pretrained"]),
        config_path=os.path.join(BASE_DIR, add_config["config"]),
        lora_config=add_config["lora"],
        device=device,
    )

    # Group by domain
    domain_files = defaultdict(list)
    file_labels = {}
    for fp, label in zip(all_filepaths, all_labels):
        domain_files[file_domains[fp]].append(fp)
        file_labels[fp] = label

    output_dir = os.path.join(BASE_DIR, "results", f"{args.nc}__{args.add}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.dataset}_scores.txt")

    results = []
    for domain_id in sorted(domain_files.keys()):
        fps = domain_files[domain_id]
        print(f"  D{domain_id} ({DOMAIN_NAMES.get(domain_id, '?')}): {len(fps)} samples")

        add_wrapper.apply_domain(domain_id)

        # Batch inference with ADD dataset (reloads audio at 16kHz)
        add_ds = WaveformDataset.__new__(WaveformDataset)
        add_ds.cut = 64600
        add_ds.items = [{"filepath": fp, "label": file_labels[fp]} for fp in fps]
        add_loader = DataLoader(add_ds, batch_size=64, shuffle=False, num_workers=4)

        with torch.no_grad():
            for audio_batch, fp_batch, _ in add_loader:
                scores = add_wrapper.predict(audio_batch)
                for i in range(len(fp_batch)):
                    results.append({
                        "filepath": fp_batch[i],
                        "label": file_labels[fp_batch[i]],
                        "spoof_score": scores[i, 0].item(),
                        "bonafide_score": scores[i, 1].item(),
                    })

    # Save scores
    with open(output_path, "w") as f:
        for r in results:
            f.write(f"{r['filepath']} {r['spoof_score']} {r['bonafide_score']}\n")
    print(f"\n[INFO] Scores saved: {output_path}")

    # ── Compute EER ──
    score_df = pd.DataFrame(results)
    bf = score_df[score_df["label"] == "bonafide"]["bonafide_score"].values
    sp = score_df[score_df["label"] == "spoof"]["bonafide_score"].values

    if len(bf) > 0 and len(sp) > 0:
        eer, _ = compute_eer(bf, sp)
        overall_eer = eer * 100
    else:
        overall_eer = None

    print(f"\n{'='*60}")
    print(f" Results: {args.nc} + {args.add} | {args.dataset}")
    print(f"  Overall EER: {overall_eer:.4f}%" if overall_eer else "  Overall EER: -")
    print(f"  Bonafide: {len(bf)} | Spoof: {len(sp)}")
    print(f"{'='*60}")

    # Save EER
    eer_path = os.path.join(output_dir, f"{args.dataset}_eer.txt")
    with open(eer_path, "w") as f:
        f.write(f"overall\t{overall_eer:.4f}\n" if overall_eer is not None else "overall\t-\n")
    print(f"EER saved: {eer_path}")


if __name__ == "__main__":
    main()
