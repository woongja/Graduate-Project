import torch
import torchaudio
from torch.utils.data import Dataset
from datautils.audio_io import load_audio

LABEL_LIST = [
    'clean',            # 0
    'background_noise', # 1
    'background_music', # 2
    'gaussian_noise',   # 3
    'band_pass_filter', # 4
    'echo',             # 5
    'pitch_shift',      # 6
    'time_stretch',     # 7
    'reverberation',    # 8
    'auto_tune',        # 9
]
LABEL2IDX  = {l: i for i, l in enumerate(LABEL_LIST)}
NUM_CLASSES = len(LABEL_LIST)

TARGET_SR = 16000   # HuBERT 입력 샘플레이트


def load_protocol(protocol_file, split=None):
    samples = []
    with open(protocol_file, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.rsplit(' ', 2)
            if len(parts) != 3:
                continue
            file_path, subset, label = parts
            if split is not None and subset != split:
                continue
            samples.append((file_path, label))
    return samples


class HubertDataset(Dataset):
    """
    HuBERT용 데이터셋. 우리가 구축한 noise classification 데이터셋 사용.
    프로토콜 파일(train_protocol.txt / eval_protocol.txt)에서 로드.

    HuBERT는 raw waveform (B, num_samples) @ 16 kHz 를 입력으로 받음.

    Returns:
        waveform  : FloatTensor (num_samples,) — 16kHz mono
        label_idx : int
    """

    def __init__(self, protocol_file, split=None,
                 clip_duration=10.0,
                 target_sr=TARGET_SR,
                 is_train=True):
        self.samples      = load_protocol(protocol_file, split=split)
        self.clip_samples = int(clip_duration * target_sr)
        self.target_sr    = target_sr
        self.is_train     = is_train

        if not self.samples:
            raise ValueError(
                f'No samples loaded from {protocol_file} (split={split}). '
                'Check the file path and split name.'
            )

        unknown = {lbl for _, lbl in self.samples} - set(LABEL2IDX)
        if unknown:
            raise ValueError(f'Unknown labels in protocol: {unknown}')

        from collections import Counter
        counts = Counter(lbl for _, lbl in self.samples)
        tag = split if split else 'all'
        print(f'[HubertDataset/{tag}] {len(self.samples)} samples  |  classes: {NUM_CLASSES}')
        for lbl in LABEL_LIST:
            print(f'  {lbl}: {counts.get(lbl, 0)}')

    def __len__(self):
        return len(self.samples)

    def _load_waveform(self, path):
        waveform, sr = load_audio(path)

        # mono 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 16kHz 리샘플링
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        waveform = waveform.squeeze(0)  # (num_samples,)

        # 길이 맞추기
        n = waveform.shape[0]
        if n < self.clip_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_samples - n))
        elif n > self.clip_samples:
            start = torch.randint(0, n - self.clip_samples + 1, (1,)).item() \
                    if self.is_train else 0
            waveform = waveform[start:start + self.clip_samples]

        return waveform  # (clip_samples,)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        waveform = self._load_waveform(file_path)
        return waveform, LABEL2IDX[label]
