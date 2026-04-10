import torch
import torchaudio
from torch.utils.data import Dataset

# 10-class label mapping (band_pass_filter = high_pass + low_pass 통합)
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
LABEL2IDX = {l: i for i, l in enumerate(LABEL_LIST)}

# high_pass_filter와 low_pass_filter를 band_pass_filter로 매핑
LABEL2IDX['high_pass_filter'] = LABEL2IDX['band_pass_filter']
LABEL2IDX['low_pass_filter'] = LABEL2IDX['band_pass_filter']

NUM_CLASSES = len(LABEL_LIST)

TARGET_SR = 32000  # HTSAT 입력 샘플레이트


def load_protocol(protocol_file, split=None):
    """
    프로토콜 파일 파싱: 각 줄 = `file_path subset label`
    파일 경로에 공백이 있을 수 있으므로 오른쪽에서 2번 split.
    """
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


class HTSATDataset(Dataset):
    """
    HTSAT용 데이터셋. 우리가 구축한 noise classification 데이터셋 사용.
    프로토콜 파일(train_protocol.txt / eval_protocol.txt)에서 로드.

    HTSAT는 raw waveform (B, num_samples) @ 32 kHz 를 입력으로 받음.

    Returns:
        waveform  : FloatTensor (num_samples,)  — 32kHz mono
        label_idx : int
    """

    def __init__(self, protocol_file, split=None,
                 clip_duration=10.0,
                 target_sr=TARGET_SR,
                 is_train=True):
        """
        Args:
            protocol_file  : 프로토콜 파일 경로
            split          : 'train' | 'dev' | None (전체)
            clip_duration  : 오디오 클립 길이 (초). 부족하면 패딩, 넘으면 랜덤 크롭(train) / 앞 크롭(eval)
            target_sr      : 리샘플 타깃 샘플레이트 (기본 32000)
            is_train       : True면 랜덤 크롭
        """
        self.samples       = load_protocol(protocol_file, split=split)
        self.clip_samples  = int(clip_duration * target_sr)
        self.target_sr     = target_sr
        self.is_train      = is_train

        if not self.samples:
            raise ValueError(
                f'No samples loaded from {protocol_file} (split={split}). '
                'Check the file path and split name.'
            )

        unknown = {lbl for _, lbl in self.samples} - set(LABEL2IDX)
        if unknown:
            raise ValueError(f'Unknown labels in protocol: {unknown}')

        from collections import Counter
        # 원본 레이블 카운트
        original_counts = Counter(lbl for _, lbl in self.samples)
        # 매핑된 레이블 카운트
        mapped_counts = Counter(LABEL2IDX[lbl] for _, lbl in self.samples)

        tag = split if split else 'all'
        print(f'[HTSATDataset/{tag}] {len(self.samples)} samples  |  classes: {NUM_CLASSES}')
        for i, lbl in enumerate(LABEL_LIST):
            count = mapped_counts.get(i, 0)
            # high_pass_filter와 low_pass_filter가 매핑된 경우 표시
            if lbl == 'band_pass_filter':
                hp_count = original_counts.get('high_pass_filter', 0)
                lp_count = original_counts.get('low_pass_filter', 0)
                if hp_count > 0 or lp_count > 0:
                    print(f'  {lbl}: {count} (high_pass: {hp_count}, low_pass: {lp_count})')
                else:
                    print(f'  {lbl}: {count}')
            else:
                print(f'  {lbl}: {count}')

    def __len__(self):
        return len(self.samples)

    def _load_waveform(self, path):
        waveform, sr = torchaudio.load(path)

        # mono 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 리샘플링
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        waveform = waveform.squeeze(0)  # (num_samples,)

        # 길이 맞추기
        n = waveform.shape[0]
        if n < self.clip_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_samples - n))
        elif n > self.clip_samples:
            if self.is_train:
                start = torch.randint(0, n - self.clip_samples + 1, (1,)).item()
            else:
                start = 0
            waveform = waveform[start:start + self.clip_samples]

        return waveform  # (clip_samples,)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        waveform = self._load_waveform(file_path)
        return waveform, LABEL2IDX[label]
