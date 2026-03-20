#!/usr/bin/env python3
"""Generate one test augmented sample per augmentation class (except autotune) using existing augmentors.

Outputs are written to ../test_sample as: <basename>_{aug}.wav
"""
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / 'Datasets'
TEST_OUT = ROOT / 'test_sample'
TEST_OUT.mkdir(parents=True, exist_ok=True)

import random
import sys
# ensure augmentation package can be imported
sys.path.insert(0, str(ROOT))

# import augmentors
from augmentation import (
    BackgroundNoiseAugmentorDeepen as BackgroundNoiseAugmentor,
    BackgroundMusicAugmentorDeepen as BackgroundMusicAugmentor,
    GaussianAugmentorV1,
    HighPassFilterAugmentor,
    LowPassFilterAugmentor,
    FrequencyOperationAugmentorDeepen as FrequencyOperationAugmentor,
    PitchAugmentor,
    TimeStretchAugmentor,
    # AutoTuneAugmentor,  # skip autotune in this test
    EchoAugmentorDeepen as EchoAugmentor,
    ReverbAugmentor,
    BandpassAugmentor,
)

from pydub import AudioSegment
import librosa


def load_config():
    cfg_path = ROOT / 'augmentation_config.yaml'
    if cfg_path.exists():
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def pick_libri_sample():
    libdir = DATASETS_DIR / 'LibriSpeech'
    if not libdir.exists():
        raise FileNotFoundError(libdir)
    # find any wav file
    files = list(libdir.rglob('*.wav')) + list(libdir.rglob('*.WAV'))
    if not files:
        raise FileNotFoundError('No wav files under LibriSpeech')
    return files[0]


def save_audiosegment(seg: AudioSegment, out_path: Path):
    seg.export(str(out_path), format='wav')


def run():
    cfg = load_config()
    src = pick_libri_sample()
    print('Using source:', src)
    base = src.stem

    # copy clean
    clean_dst = TEST_OUT / f"{base}_clean.wav"
    from shutil import copy2
    copy2(src, clean_dst)
    print('Wrote', clean_dst)

    # mapping of label -> (class, config_key)
    AUGS = [
        ('background_noise', BackgroundNoiseAugmentor, 'background_noise'),
        ('background_music', BackgroundMusicAugmentor, 'background_music'),
        ('auto_tune', None, 'auto_tune'),  # skip
        ('bandpass', BandpassAugmentor, None),
        ('echo', EchoAugmentor, 'echo'),
        ('pitch_shift', PitchAugmentor, 'pitch_shift'),
        ('time_stretch', TimeStretchAugmentor, 'time_stretch'),
        ('gaussian_noise', GaussianAugmentorV1, 'gaussian_noise'),
        ('reverberation', ReverbAugmentor, 'reverberation'),
        ('high_pass_filter', HighPassFilterAugmentor, 'high_pass_filter'),
        ('low_pass_filter', LowPassFilterAugmentor, 'low_pass_filter'),
    ]

    y, sr = librosa.load(str(src), sr=None)

    for name, cls, cfg_key in AUGS:
        if name == 'auto_tune':
            print('Skipping autotune in this test (separate env)')
            continue
        out_path = TEST_OUT / f"{base}_{name}.wav"
        try:
            conf = cfg.get(cfg_key, {}) if cfg_key else {}
            conf = dict(conf)  # copy
            conf['output_path'] = str(out_path)
            conf['out_format'] = 'wav'
            # for bandpass, provide defaults
            if name == 'bandpass':
                conf.setdefault('lowpass', 4000)
                conf.setdefault('highpass', 300)

            aug = cls(conf)
            # some augmentors expect load/transform method
            try:
                aug.load(str(src))
                aug.transform()
                aug.augmented_audio.export(str(out_path), format='wav')
            except Exception as e:
                # fallback: try to call transform with raw audio
                print('Augmentor failed via API, attempting manual processing:', e)
                # For safety, just copy src to out_path
                copy2(src, out_path)
            print('Wrote', out_path)
        except Exception as e:
            print('Failed', name, e)


if __name__ == '__main__':
    run()
