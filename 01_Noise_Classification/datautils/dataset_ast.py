import torch
import torchaudio
from torch.utils.data import Dataset
from datautils.audio_io import load_audio

# aug_type → integer label mapping (10 classes)
# high_pass_filter and low_pass_filter are merged as band_pass_filter
LABEL_LIST = [
    'clean',            # 0
    'background_noise', # 1
    'background_music', # 2
    'gaussian_noise',   # 3
    'band_pass_filter', # 4  (high_pass_filter + low_pass_filter)
    'echo',             # 5
    'pitch_shift',      # 6
    'time_stretch',     # 7
    'reverberation',    # 8
    'auto_tune',        # 9
]
LABEL2IDX = {l: i for i, l in enumerate(LABEL_LIST)}
NUM_CLASSES = len(LABEL_LIST)


def load_protocol(protocol_file, split=None):
    """
    Parse protocol file: each line is  `file_path subset label`
    File path may contain spaces, so split from the right.

    Args:
        protocol_file : path to train_protocol.txt or eval_protocol.txt
        split         : if given, keep only rows matching this subset value
                        (e.g. 'train', 'dev').  None = keep all rows.
    Returns:
        list of (file_path, label) tuples
    """
    samples = []
    with open(protocol_file, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            # split from the right to handle spaces in file paths
            parts = line.rsplit(' ', 2)
            if len(parts) != 3:
                continue
            file_path, subset, label = parts
            if split is not None and subset != split:
                continue
            samples.append((file_path, label))
    return samples


class ASTDataset(Dataset):
    """
    Dataset for AST noise classification.
    Reads from protocol files produced by 0_preprocess.ipynb.

    Protocol format (space-separated, no header):
        file_path  subset  label

    Returns: (fbank, label_idx)
        fbank     : FloatTensor (target_length, 128)
        label_idx : int
    """

    def __init__(self, protocol_file, split=None,
                 target_length=512,
                 freqm=48, timem=192,
                 mean=-4.2677393, std=4.5689974,
                 is_train=True):
        self.samples = load_protocol(protocol_file, split=split)
        if not self.samples:
            raise ValueError(
                f'No samples loaded from {protocol_file} '
                f'(split={split}). Check the file path and split name.'
            )

        self.target_length = target_length
        self.freqm = freqm if is_train else 0
        self.timem = timem if is_train else 0
        self.mean  = mean
        self.std   = std

        unknown = {lbl for _, lbl in self.samples} - set(LABEL2IDX)
        if unknown:
            raise ValueError(f'Unknown labels in protocol: {unknown}')

        from collections import Counter
        counts = Counter(lbl for _, lbl in self.samples)
        tag = split if split else 'all'
        print(f'[ASTDataset/{tag}] {len(self.samples)} samples  |  classes: {NUM_CLASSES}')
        for lbl in LABEL_LIST:
            print(f'  {lbl}: {counts.get(lbl, 0)}')

    def __len__(self):
        return len(self.samples)

    def _load_fbank(self, path):
        waveform, sr = load_audio(path)
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )  # (n_frames, 128)

        p = self.target_length - fbank.shape[0]
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        elif p < 0:
            fbank = fbank[:self.target_length]
        return fbank  # (target_length, 128)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        fbank = self._load_fbank(file_path)  # (T, 128)

        # SpecAugment (train only)
        if self.freqm > 0 or self.timem > 0:
            fbank = fbank.unsqueeze(0).transpose(1, 2)  # (1, 128, T)
            if self.freqm > 0:
                fbank = torchaudio.transforms.FrequencyMasking(self.freqm)(fbank)
            if self.timem > 0:
                fbank = torchaudio.transforms.TimeMasking(self.timem)(fbank)
            fbank = fbank.squeeze(0).transpose(0, 1)  # (T, 128)

        fbank = (fbank - self.mean) / (self.std * 2)
        return fbank, LABEL2IDX[label]
