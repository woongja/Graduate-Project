#!/usr/bin/env python3
"""
Merge NC_DF CSV files into protocol files for LoRA training

Input: CSV files in NC_DF directory (per noise type)
Output: Merged protocol files (asv19_train.txt, asv19_dev.txt, asv19_eval.txt, df21_eval.txt)
Format: file_path label1 aug_type (space-separated)
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
NC_DF_DIR = Path("/home/woongjae/ADD_LAB/projects/Graduate-Project/00_Dataset_gen/Datasets/noise_dataset/NC_DF")
OUTPUT_DIR = Path("/home/woongjae/ADD_LAB/projects/Graduate-Project/02_LoRA_Training/protocols")

# Noise types (11 types)
NOISE_TYPES = [
    'auto_tune',
    'background_music',
    'background_noise',
    'clean',
    'echo',
    'gaussian_noise',
    'high_pass_filter',
    'low_pass_filter',
    'pitch_shift',
    'reverberation',
    'time_stretch',
]


def merge_protocols(dataset, split):
    """
    Merge CSV files for a given dataset and split

    Args:
        dataset: 'asv19' or 'df21'
        split: 'train', 'dev', or 'eval'

    Returns:
        DataFrame with merged data
    """
    print(f"\n>>> Merging {dataset}_{split}...")

    dfs = []
    for noise_type in NOISE_TYPES:
        csv_path = NC_DF_DIR / f"metadata_{dataset}_{split}_{noise_type}.csv"

        if not csv_path.exists():
            print(f"  [WARNING] File not found: {csv_path.name}")
            continue

        # Read CSV
        df = pd.read_csv(csv_path)

        # Select only needed columns: file_path, label1, aug_type
        df_selected = df[['file_path', 'label1', 'aug_type']].copy()

        dfs.append(df_selected)
        print(f"  ✓ {noise_type:20s} : {len(df_selected):7,} samples")

    if not dfs:
        print(f"  [ERROR] No CSV files found for {dataset}_{split}")
        return None

    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    print(f"  → Total: {len(merged_df):,} samples")

    return merged_df


def save_protocol(df, output_path):
    """
    Save protocol file in space-separated format

    Args:
        df: DataFrame with columns [file_path, label1, aug_type]
        output_path: Path to save the protocol file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as space-separated txt (no header)
    with open(output_path, 'w') as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {output_path.name}"):
            f.write(f"{row['file_path']} {row['label1']} {row['aug_type']}\n")

    print(f"  ✓ Saved: {output_path} ({len(df):,} lines)")


def main():
    print("=" * 60)
    print("NC_DF Protocol Merger")
    print("=" * 60)

    # ASV19: train, dev, eval
    for split in ['train', 'dev', 'eval']:
        df = merge_protocols('asv19', split)
        if df is not None:
            output_path = OUTPUT_DIR / f"asv19_{split}.txt"
            save_protocol(df, output_path)

    # DF21: eval only
    df = merge_protocols('df21', 'eval')
    if df is not None:
        output_path = OUTPUT_DIR / "df21_eval.txt"
        save_protocol(df, output_path)

    print("\n" + "=" * 60)
    print("Protocol merging completed!")
    print("=" * 60)

    # Summary
    print("\nGenerated files:")
    for fname in ['asv19_train.txt', 'asv19_dev.txt', 'asv19_eval.txt', 'df21_eval.txt']:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            num_lines = sum(1 for _ in open(fpath))
            print(f"  • {fname:20s} : {num_lines:7,} samples")
        else:
            print(f"  • {fname:20s} : NOT FOUND")


if __name__ == "__main__":
    main()
