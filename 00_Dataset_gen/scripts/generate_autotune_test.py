#!/usr/bin/env python3
"""
Create one autotune test sample from a LibriSpeech WAV.

Usage (run in autotune conda env):
  python3 00_Dataset_gen/scripts/generate_autotune_test.py

The script picks the first WAV under `00_Dataset_gen/Datasets/LibriSpeech` (or /nvme3 path),
runs the AutoTuneAugmentor, and writes output to `00_Dataset_gen/test_sample/<basename>_auto_tune.wav`.
It also disables numba JIT inside the process to avoid numba typing/JIT issues.
"""
from pathlib import Path
import os
import sys
import traceback

# Disable numba JIT at runtime to avoid compilation errors in environments where numba
# and llvmlite versions mismatch. This should be set before importing libraries that
# may use numba.
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / 'Datasets'
OUT_DIR = ROOT / 'test_sample'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_libri_sample():
    paths = [DATASETS_DIR / 'LibriSpeech', Path('/nvme3/Datasets/WJ/LibriSpeech'), Path('/nvme1/Datasets/WJ/LibriSpeech')]
    for p in paths:
        if p and p.exists():
            for ext in ('*.wav', '*.WAV', '*.flac'):
                files = list(p.rglob(ext))
                if files:
                    return files[0]
    return None


def main():
    sample = find_libri_sample()
    if sample is None:
        print('No LibriSpeech sample found under expected paths.')
        sys.exit(1)
    print('Using sample:', sample)

    # Make augmentation package importable
    sys.path.insert(0, str(ROOT / '00_Dataset_gen'))
    # If augmentation package is at ROOT/00_Dataset_gen/augmentation, add that parent
    sys.path.insert(0, str(ROOT))

    try:
        # Import after NUMBA_DISABLE_JIT set
        from augmentation import AutoTuneAugmentor
    except Exception as e:
        print('Failed to import AutoTuneAugmentor:', e)
        traceback.print_exc()
        sys.exit(1)

    out_path = OUT_DIR / f"{sample.stem}_auto_tune.wav"
    try:
        # Provide required BaseAugmentor config fields: output_path and out_format
        conf = {
            'aug_type': 'auto_tune',
            'correction_method': 'closest',
            'output_path': str(OUT_DIR),
            'out_format': 'wav',
        }
        aug = AutoTuneAugmentor(conf)
        aug.load(str(sample))
        aug.transform()

        # augmented_audio is expected to be a pydub.AudioSegment
        try:
            aug.augmented_audio.export(str(out_path), format='wav')
        except Exception:
            # fallback: if it's numpy array-like, use soundfile
            try:
                import soundfile as sf
                import numpy as np
                data = np.array(aug.augmented_audio)
                sf.write(str(out_path), data, int(aug.sr))
            except Exception as e:
                print('Failed to save augmented audio:', e)
                traceback.print_exc()
                sys.exit(1)

        print('Wrote autotune sample to', out_path)
    except Exception as e:
        print('Autotune processing failed:')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
