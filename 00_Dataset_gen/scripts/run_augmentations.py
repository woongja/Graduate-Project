#!/usr/bin/env python3
"""
Run augmentations for files listed in a protocol CSV.

Assignment strategy:
- 10 classes total, each gets N/10 of the files (1:1 balanced)
- bandpass is one logical class split internally: N/20 high_pass + N/20 low_pass
- clean: no augmentation applied, original file path recorded as-is
- auto_tune: separate conda env → handled via --only-autotune flag

Class list:
  background_noise, background_music, bandpass (high/low), echo,
  pitch_shift, time_stretch, gaussian_noise, reverberation, clean, auto_tune

Output:
  {output_base}/{split}/{aug_type}/{speaker}_{utt}__{aug_type}.wav
  {output_base}/metadata_{split}_{aug_type}.csv
"""
import argparse
import csv
import logging
import random
from pathlib import Path
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from tqdm import tqdm

RANDOM_SEED = 42

# 10 logical classes → actual aug_type assignments
# (logical_class, aug_type, relative_weight)
# bandpass = high_pass(0.5) + low_pass(0.5) → together = 1 unit
CLASS_ASSIGNMENTS = [
    ("background_noise", "background_noise", 1),
    ("background_music", "background_music", 1),
    ("bandpass",         "high_pass_filter", 0.5),
    ("bandpass",         "low_pass_filter",  0.5),
    ("echo",             "echo",             1),
    ("pitch_shift",      "pitch_shift",      1),
    ("time_stretch",     "time_stretch",     1),
    ("gaussian_noise",   "gaussian_noise",   1),
    ("reverberation",    "reverberation",    1),
    ("clean",            "clean",            1),
    ("auto_tune",        "auto_tune",        1),
]
TOTAL_UNITS = 10  # 10 logical classes


def assign_aug_types(rows, seed=RANDOM_SEED):
    """Shuffle rows and assign one aug_type per file to ensure 1:1 class balance."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    N = len(rows)

    # Build assignment list: each entry is aug_type for one file slot
    assignments = []
    for _, aug_type, weight in CLASS_ASSIGNMENTS:
        count = round(N * weight / TOTAL_UNITS)
        assignments.extend([aug_type] * count)

    # Adjust length to exactly match N (rounding may cause ±1 difference)
    while len(assignments) < N:
        assignments.append(assignments[-1])
    assignments = assignments[:N]

    rng.shuffle(assignments)  # shuffle so aug types aren't block-sequential

    return [(r, aug_t) for r, aug_t in zip(rows, assignments)]


def find_augmentor_class(aug_module, aug_type_name):
    norm = aug_type_name.replace('_', '').lower()
    candidates = []
    for attr in dir(aug_module):
        if not attr.endswith('Augmentor') or attr.startswith('_'):
            continue
        n = attr.replace('Augmentor', '').lower()
        if norm in n or n in norm:
            candidates.append(attr)
    if not candidates:
        return None
    candidates.sort(key=lambda x: len(x))
    return getattr(aug_module, candidates[0])


def short_param_from_conf(conf):
    keys = ['snr', 'rt60', 'rir_file', 'delay', 'decay', 'min_semitones', 'max_semitones']
    parts = [f"{k}={conf[k]}" for k in keys if k in conf]
    return ','.join(parts)[:120].replace(' ', '')


def run_for_protocol(protocol_csv, aug_configs, output_base, split, only_autotune=False):
    import augmentation as augmod
    output_base = Path(output_base)

    with open(protocol_csv) as f:
        rows = list(csv.DictReader(f))

    assigned = assign_aug_types(rows)

    # Filter by mode
    if only_autotune:
        assigned = [(r, t) for r, t in assigned if t == 'auto_tune']
    else:
        assigned = [(r, t) for r, t in assigned if t != 'auto_tune']

    results_by_aug = defaultdict(list)

    for r, aug_type in tqdm(assigned, desc=f'[{split}]'):
        in_path = Path(r.get('file_path', '')).expanduser()
        if not in_path.exists():
            logging.warning('Missing input: %s', in_path)
            continue

        speaker = r.get('speaker', '').strip()
        utt     = r.get('utt', in_path.stem).strip()
        label1  = r.get('label1', '').strip()
        dataset = r.get('dataset', '').strip()
        out_basename_base = f"{speaker}_{utt}"

        aug_out_dir = output_base / split / aug_type
        aug_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = aug_out_dir / f"{out_basename_base}__{aug_type}.wav"

        # clean: copy original file to output dir
        if aug_type == 'clean':
            import shutil
            shutil.copy2(str(in_path), str(out_path))
            results_by_aug[aug_type].append({
                'file_path': str(out_path.resolve()),
                'speaker':   speaker,
                'utt':       utt,
                'dataset':   dataset,
                'label1':    label1,
                'aug_type':  aug_type,
                'split':     split,
            })
            continue

        conf = aug_configs.get(aug_type)
        if conf is None:
            logging.warning('No config found for aug_type: %s', aug_type)
            continue

        cls = find_augmentor_class(augmod, aug_type)
        if cls is None:
            logging.warning('No augmentor class found for: %s', aug_type)
            continue

        local_conf = dict(conf)
        local_conf['aug_type']    = aug_type
        local_conf['output_path'] = str(aug_out_dir)
        local_conf['out_format']  = 'wav'
        local_conf['speaker']     = speaker
        local_conf['utt']         = utt

        try:
            aug = cls(local_conf)
            aug.load(str(in_path))
            aug.transform()

            try:
                if hasattr(aug, 'augmented_audio') and aug.augmented_audio is not None:
                    aug.augmented_audio.export(str(out_path), format='wav')
                else:
                    aug.save()
                    out_path = Path(local_conf['output_path']) / f"{aug.file_name}.{local_conf['out_format']}"
            except Exception:
                import numpy as np
                import soundfile as sf
                data = getattr(aug, 'augmented_audio', None) or getattr(aug, 'data', None)
                if isinstance(data, np.ndarray):
                    sf.write(str(out_path), data, int(getattr(aug, 'sr', 16000)))
                else:
                    raise

            results_by_aug[aug_type].append({
                'file_path': str(out_path.resolve()),
                'speaker':   speaker,
                'utt':       utt,
                'dataset':   dataset,
                'label1':    label1,
                'aug_type':  aug_type,
                'split':     split,
            })
        except Exception as e:
            logging.exception('Augmentation failed for %s with %s: %s', in_path, aug_type, e)

    for aug_type, items in results_by_aug.items():
        out_csv = output_base / f"metadata_{split}_{aug_type}.csv"
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['file_path', 'speaker', 'utt', 'dataset', 'label1', 'aug_type', 'split']
            )
            writer.writeheader()
            writer.writerows(items)
        print(f"Metadata saved: {out_csv} ({len(items)} entries)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol',      required=True)
    parser.add_argument('--split',         required=True, help='train / dev / eval')
    parser.add_argument('--config',        default=str(ROOT / 'augmentation_config.yaml'))
    parser.add_argument('--output-base',   default=str(ROOT / 'Datasets' / 'noise_dataset' / 'augmented'))
    parser.add_argument('--only-autotune', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    run_for_protocol(args.protocol, conf, args.output_base, args.split,
                     only_autotune=args.only_autotune)


if __name__ == '__main__':
    main()
