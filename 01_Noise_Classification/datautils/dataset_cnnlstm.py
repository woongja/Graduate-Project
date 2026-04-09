"""
CNNLSTMDataset — mel spectrogram dataset for CNNLSTMClassifier
  - 출력: (1, n_mels, T) mel spectrogram, label_idx
  - sample_rate: 22050 Hz
  - n_mels: 128, n_fft: 1024, hop_length: 512
  - 10초 기준 T ≈ 431 프레임
"""
import random
import torch
import torchaudio
from datautils.audio_io import load_audio
import torchaudio.transforms as T
from torch.utils.data import Dataset

LABEL_LIST = [
    'clean', 'background_noise', 'background_music', 'gaussian_noise',
    'band_pass_filter', 'echo', 'pitch_shift', 'time_stretch',
    'reverberation', 'auto_tune',
]
LABEL2IDX = {l: i for i, l in enumerate(LABEL_LIST)}

TARGET_SR   = 22050
N_MELS      = 128
N_FFT       = 1024
HOP_LENGTH  = 512


class CNNLSTMDataset(Dataset):
    def __init__(self, protocol_path: str, split=None,
                 clip_duration: float = 10.0, is_train: bool = False):
        """
        Args:
            protocol_path: 'file_path subset label' 형식 텍스트 파일
            split:         'train' | 'dev' | None (eval 전체)
            clip_duration: 오디오 클립 길이 (초)
            is_train:      True면 랜덤 크롭, False면 앞부분 크롭
        """
        self.clip_samples = int(clip_duration * TARGET_SR)
        self.is_train     = is_train

        self.mel_transform = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

        self.samples = []
        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().rsplit(' ', 2)
                if len(parts) != 3:
                    continue
                path, subset, label = parts
                if split is not None and subset != split:
                    continue
                if label not in LABEL2IDX:
                    continue
                self.samples.append((path, LABEL2IDX[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        waveform, sr = load_audio(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

        # 길이 맞추기
        n = waveform.shape[1]
        if n >= self.clip_samples:
            if self.is_train:
                start = random.randint(0, n - self.clip_samples)
            else:
                start = 0
            waveform = waveform[:, start: start + self.clip_samples]
        else:
            pad = self.clip_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # mel spectrogram (1, n_mels, T)
        spec = self.mel_transform(waveform)      # (1, n_mels, T)
        spec = self.amplitude_to_db(spec)        # dB 스케일

        return spec, label
