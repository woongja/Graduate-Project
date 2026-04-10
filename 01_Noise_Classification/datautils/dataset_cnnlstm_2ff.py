"""
CNNLSTM_2FF Dataset — Spectrogram + F0 for 2-Feature Fusion
  - 출력: (spec, f0, label)
    - spec: (1, n_mels, T) mel spectrogram
    - f0: (1, T_f0) fundamental frequency contour
  - sample_rate: 22050 Hz
  - n_mels: 128, n_fft: 1024, hop_length: 512
"""
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    print("[Warning] CREPE not available. Install: pip install crepe")


def load_audio(path):
    """Load audio file using torchaudio"""
    waveform, sr = torchaudio.load(path)
    return waveform, sr


LABEL_LIST = [
    'clean', 'background_noise', 'background_music', 'gaussian_noise',
    'band_pass_filter', 'echo', 'pitch_shift', 'time_stretch',
    'reverberation', 'auto_tune',
]
LABEL2IDX = {l: i for i, l in enumerate(LABEL_LIST)}

# high_pass_filter와 low_pass_filter를 band_pass_filter로 매핑
LABEL2IDX['high_pass_filter'] = LABEL2IDX['band_pass_filter']
LABEL2IDX['low_pass_filter'] = LABEL2IDX['band_pass_filter']

TARGET_SR   = 22050
N_MELS      = 128
N_FFT       = 1024
HOP_LENGTH  = 512


class CNNLSTM_2FF_Dataset(Dataset):
    """Dataset for 2-Feature Fusion: Spectrogram + F0"""

    def __init__(
        self,
        protocol_path: str,
        split=None,
        clip_duration: float = 10.0,
        is_train: bool = False,
        f0_method: str = 'crepe'  # 'crepe' or 'yin'
    ):
        """
        Args:
            protocol_path: 'file_path subset label' 형식 텍스트 파일
            split:         'train' | 'dev' | None (eval 전체)
            clip_duration: 오디오 클립 길이 (초)
            is_train:      True면 랜덤 크롭, False면 앞부분 크롭
            f0_method:     'crepe' or 'yin' (F0 추출 방법)
        """
        self.clip_samples = int(clip_duration * TARGET_SR)
        self.is_train     = is_train
        self.f0_method    = f0_method

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

        # Load protocol
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

        print(f"[CNNLSTM_2FF_Dataset] Loaded {len(self.samples)} samples (split={split})")

    def __len__(self):
        return len(self.samples)

    def extract_f0_crepe(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract F0 using CREPE

        Args:
            waveform: (1, samples) tensor
        Returns:
            f0: (1, T_f0) normalized F0 contour
        """
        if not CREPE_AVAILABLE:
            # Fallback: return zeros
            T_f0 = (waveform.shape[1] // HOP_LENGTH) + 1
            return torch.zeros(1, T_f0)

        audio_np = waveform.squeeze(0).numpy()

        # CREPE expects sampling rate and audio as numpy array
        time, frequency, confidence, activation = crepe.predict(
            audio_np,
            TARGET_SR,
            viterbi=True,
            step_size=HOP_LENGTH / TARGET_SR * 1000  # ms
        )

        # Filter unvoiced frames (confidence < 0.5)
        f0 = frequency.copy()
        f0[confidence < 0.5] = 0.0

        # Normalize F0 (log scale)
        f0_norm = np.where(f0 > 0, np.log(f0 + 1e-8), 0.0)
        f0_norm = (f0_norm - f0_norm.mean()) / (f0_norm.std() + 1e-8)

        f0_tensor = torch.from_numpy(f0_norm).float().unsqueeze(0)  # (1, T_f0)

        return f0_tensor

    def extract_f0_yin(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract F0 using YIN algorithm (torchaudio)

        Args:
            waveform: (1, samples) tensor
        Returns:
            f0: (1, T_f0) normalized F0 contour
        """
        # Use torchaudio's sliding_window_cmn or manual implementation
        # For simplicity, we'll use a basic approach
        frame_length = N_FFT
        hop_length = HOP_LENGTH

        # Simple autocorrelation-based pitch detection
        # This is a placeholder - you can implement YIN properly
        waveform_np = waveform.squeeze(0).numpy()
        n_frames = (len(waveform_np) - frame_length) // hop_length + 1

        f0_values = []
        for i in range(n_frames):
            start = i * hop_length
            frame = waveform_np[start:start + frame_length]

            # Simple autocorrelation peak detection
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find first peak after lag
            min_lag = int(TARGET_SR / 500)  # 500 Hz max
            max_lag = int(TARGET_SR / 50)   # 50 Hz min

            if max_lag < len(autocorr):
                peaks = autocorr[min_lag:max_lag]
                if len(peaks) > 0:
                    lag = np.argmax(peaks) + min_lag
                    f0 = TARGET_SR / lag if lag > 0 else 0.0
                else:
                    f0 = 0.0
            else:
                f0 = 0.0

            f0_values.append(f0)

        f0_np = np.array(f0_values)
        f0_norm = np.where(f0_np > 0, np.log(f0_np + 1e-8), 0.0)
        f0_norm = (f0_norm - f0_norm.mean()) / (f0_norm.std() + 1e-8)

        f0_tensor = torch.from_numpy(f0_norm).float().unsqueeze(0)

        return f0_tensor

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load audio
        waveform, sr = load_audio(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

        # Clip to duration
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

        # Extract Mel Spectrogram
        spec = self.mel_transform(waveform)      # (1, n_mels, T)
        spec = self.amplitude_to_db(spec)        # dB scale

        # Extract F0
        if self.f0_method == 'crepe':
            f0 = self.extract_f0_crepe(waveform)
        else:
            f0 = self.extract_f0_yin(waveform)

        # Align F0 length with spectrogram time dimension
        T_spec = spec.shape[2]
        T_f0 = f0.shape[1]

        if T_f0 != T_spec:
            # Interpolate F0 to match spectrogram time dimension
            f0 = torch.nn.functional.interpolate(
                f0.unsqueeze(0),  # (1, 1, T_f0)
                size=T_spec,
                mode='linear',
                align_corners=False
            ).squeeze(0)  # (1, T_spec)

        return spec, f0, label


# Collate function for DataLoader
def collate_2ff(batch):
    """
    Custom collate function for variable-length sequences

    Args:
        batch: List of (spec, f0, label) tuples
    Returns:
        specs: (B, 1, n_mels, T_max)
        f0s: (B, 1, T_max)
        labels: (B,)
    """
    specs, f0s, labels = zip(*batch)

    # Find max time dimension
    T_max = max(s.shape[2] for s in specs)

    # Pad spectrograms
    specs_padded = []
    for spec in specs:
        T = spec.shape[2]
        if T < T_max:
            pad = T_max - T
            spec = torch.nn.functional.pad(spec, (0, pad))
        specs_padded.append(spec)

    # Pad F0
    f0s_padded = []
    for f0 in f0s:
        T = f0.shape[1]
        if T < T_max:
            pad = T_max - T
            f0 = torch.nn.functional.pad(f0, (0, pad))
        f0s_padded.append(f0)

    specs_batch = torch.stack(specs_padded)  # (B, 1, n_mels, T_max)
    f0s_batch = torch.stack(f0s_padded)      # (B, 1, T_max)
    labels_batch = torch.tensor(labels, dtype=torch.long)

    return specs_batch, f0s_batch, labels_batch


if __name__ == "__main__":
    # Test dataset
    protocol_path = "../protocols/train_protocol.txt"

    dataset = CNNLSTM_2FF_Dataset(
        protocol_path,
        split='train',
        clip_duration=10.0,
        is_train=True,
        f0_method='crepe'
    )

    print(f"Dataset size: {len(dataset)}")

    # Test one sample
    if len(dataset) > 0:
        spec, f0, label = dataset[0]
        print(f"\nSample 0:")
        print(f"  Spec shape: {spec.shape}")
        print(f"  F0 shape:   {f0.shape}")
        print(f"  Label:      {label} ({LABEL_LIST[label]})")

        # Test collate
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_2ff)
        batch = next(iter(loader))
        specs, f0s, labels = batch
        print(f"\nBatch shapes:")
        print(f"  Specs:  {specs.shape}")
        print(f"  F0s:    {f0s.shape}")
        print(f"  Labels: {labels.shape}")
