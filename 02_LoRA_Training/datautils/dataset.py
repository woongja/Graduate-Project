"""ASVspoof Dataset for Audio Deepfake Detection with LoRA Training.

Protocol format:
    file_path bonafide/spoof
or
    file_path label_int (0=bonafide, 1=spoof)
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Tuple, Optional


class ASVspoofDataset(Dataset):
    """ASVspoof dataset for bonafide/spoof classification.

    Args:
        protocol_path (str): Path to protocol file
        data_dir (str): Root directory of audio files
        sample_rate (int): Target sample rate (default: 16000)
        trim_length (int): Fixed audio length in samples (default: 64000, i.e., 4 seconds)
        augmentation (callable, optional): Augmentation function
    """

    def __init__(
        self,
        protocol_path: str,
        data_dir: str,
        sample_rate: int = 16000,
        trim_length: int = 64000,
        augmentation: Optional[callable] = None
    ):
        self.protocol_path = protocol_path
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.trim_length = trim_length
        self.augmentation = augmentation

        self.data_list = []
        self.labels = []

        self._load_protocol()

    def _load_protocol(self):
        """Load protocol file."""
        with open(self.protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                file_path = parts[0]
                label_str = parts[1]

                # Convert label to integer
                if label_str.lower() in ['bonafide', 'bona_fide', '0']:
                    label = 0
                elif label_str.lower() in ['spoof', 'fake', '1']:
                    label = 1
                else:
                    try:
                        label = int(label_str)
                    except ValueError:
                        print(f"Warning: Invalid label '{label_str}' in {file_path}, skipping...")
                        continue

                # Full path to audio file
                if not os.path.isabs(file_path):
                    file_path = os.path.join(self.data_dir, file_path)

                self.data_list.append(file_path)
                self.labels.append(label)

        print(f"Loaded {len(self.data_list)} samples from {self.protocol_path}")
        print(f"  Bonafide: {self.labels.count(0)}")
        print(f"  Spoof: {self.labels.count(1)}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            waveform (Tensor): Audio tensor of shape (trim_length,)
            label (int): 0 for bonafide, 1 for spoof
        """
        file_path = self.data_list[idx]
        label = self.labels[idx]

        # Load audio
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor if loading fails
            waveform = torch.zeros(1, self.trim_length)
            sr = self.sample_rate

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Trim or pad to fixed length
        waveform = self._trim_or_pad(waveform)

        # Apply augmentation if provided
        if self.augmentation is not None:
            waveform = self.augmentation(waveform)

        # Squeeze to (length,)
        waveform = waveform.squeeze(0)

        return waveform, label

    def _trim_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trim or pad waveform to fixed length.

        Args:
            waveform: (1, length)

        Returns:
            waveform: (1, trim_length)
        """
        current_length = waveform.shape[1]

        if current_length > self.trim_length:
            # Random crop
            start = torch.randint(0, current_length - self.trim_length + 1, (1,)).item()
            waveform = waveform[:, start:start + self.trim_length]
        elif current_length < self.trim_length:
            # Repeat padding
            num_repeats = (self.trim_length // current_length) + 1
            waveform = waveform.repeat(1, num_repeats)
            waveform = waveform[:, :self.trim_length]

        return waveform


def collate_fn(batch):
    """Collate function for DataLoader.

    Args:
        batch: List of (waveform, label) tuples

    Returns:
        waveforms: (batch_size, length)
        labels: (batch_size,)
    """
    waveforms, labels = zip(*batch)
    waveforms = torch.stack(waveforms)
    labels = torch.tensor(labels, dtype=torch.long)
    return waveforms, labels
