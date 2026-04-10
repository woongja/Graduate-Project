"""
ASVspoof Dataset Loader for Audio Deepfake Detection (Binary Classification: bonafide vs spoof)

Protocol format (ASVspoof2019/2021):
    speaker_id audio_id - system_id label
    Example: LA_T_1000032 LA_T_1000032 - - bonafide
             LA_T_1000033 LA_T_1000033 - A07 spoof

Label mapping:
    bonafide -> 0
    spoof -> 1
"""
import os
import warnings
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", message="PySoundFile failed")


class ASVspoofDataset(Dataset):
    """
    ASVspoof Dataset for binary classification (bonafide vs spoof)
    """
    # Domain -> augmentation type mapping for multi-LoRA filtering
    DOMAIN_TO_AUGTYPES = {
        0: ['clean'],
        1: ['background_music', 'background_noise'],
        2: ['auto_tune'],
        3: ['band_pass_filter', 'high_pass_filter', 'low_pass_filter'],
        4: ['echo'],
        5: ['pitch_shift', 'time_stretch'],
        6: ['gaussian_noise'],
        7: ['reverberation'],
    }

    def __init__(
        self,
        protocol_path,
        audio_dir,
        augmentation_type='none',
        sample_rate=16000,
        max_length=64600,  # 4 seconds at 16kHz
        padding_type='repeat',
        random_start=False,
        domain_filter=None
    ):
        """
        Args:
            protocol_path: Path to ASVspoof protocol file
            audio_dir: Directory containing audio files (.flac)
            augmentation_type: Augmentation type (e.g., 'none', 'clean', 'background_noise', etc.)
            sample_rate: Target sample rate (default 16000)
            max_length: Maximum audio length in samples (default 64600 = 4 seconds)
            padding_type: Padding type ('repeat' or 'zero')
            random_start: Whether to randomly crop audio
            domain_filter: Domain ID (0-7) to filter by augmentation type for multi-LoRA training.
                           None means no filtering (use all data).
        """
        self.protocol_path = protocol_path
        self.audio_dir = audio_dir
        self.augmentation_type = augmentation_type
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.padding_type = padding_type
        self.random_start = random_start
        self.domain_filter = domain_filter

        # Build allowed augmentation types for filtering
        if domain_filter is not None and domain_filter in self.DOMAIN_TO_AUGTYPES:
            self._allowed_augtypes = set(self.DOMAIN_TO_AUGTYPES[domain_filter])
        else:
            self._allowed_augtypes = None

        # Load protocol
        self.data_list = self._load_protocol()

        print(f"Loaded {len(self.data_list)} samples from {protocol_path}")
        print(f"Audio directory: {audio_dir}")
        if self._allowed_augtypes:
            print(f"Domain filter: g{domain_filter} -> {self._allowed_augtypes}")

    def _load_protocol(self):
        """
        Load protocol file

        Supports two formats:
        1. ASVspoof 5-column: speaker_id audio_id - system_id label
        2. Path-based 3-column: /path/to/audio.wav label augmentation_type

        Returns:
            data_list: List of (audio_path, label) tuples
        """
        data_list = []

        with open(self.protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                # Format 1: ASVspoof 5-column format
                if len(parts) >= 5:
                    audio_id = parts[1]
                    label_str = parts[4]  # 'bonafide' or 'spoof'
                    audio_path = None  # Will construct in __getitem__

                # Format 2: Path-based 3-column format
                elif len(parts) == 3:
                    audio_path = parts[0]  # Full path
                    label_str = parts[1]  # 'bonafide' or 'spoof'
                    aug_type = parts[2]   # augmentation type
                    audio_id = None

                    # Domain filtering for multi-LoRA training
                    if self._allowed_augtypes is not None:
                        if aug_type not in self._allowed_augtypes:
                            continue

                else:
                    continue

                # Convert label to binary: bonafide=0, spoof=1
                if label_str == 'bonafide':
                    label = 0
                elif label_str == 'spoof':
                    label = 1
                else:
                    print(f"Warning: Unknown label '{label_str}'")
                    continue

                data_list.append((audio_path if audio_path else audio_id, label))

        print(f"Loaded {len(data_list)} samples from {self.protocol_path}")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Get item by index

        Returns:
            waveform: [1, max_length] tensor
            label: 0 (bonafide) or 1 (spoof)
            audio_id: audio file ID or path
        """
        audio_info, label = self.data_list[idx]

        # Check if audio_info is a full path or just an ID
        if audio_info.startswith('/') or audio_info.startswith('.'):
            # Full path provided in protocol
            audio_path = audio_info
            audio_id = os.path.basename(audio_path)
        else:
            # Audio ID only - construct path
            audio_id = audio_info
            # Handle augmentation directory structure
            if self.augmentation_type == 'none' or self.augmentation_type == 'clean':
                audio_path = os.path.join(self.audio_dir, f"{audio_id}.flac")
            else:
                # For augmented data: audio_dir/augmentation_type/audio_id.flac
                audio_path = os.path.join(self.audio_dir, self.augmentation_type, f"{audio_id}.flac")

        # Load audio with librosa
        try:
            # librosa.load automatically resamples to sr and converts to mono
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Convert numpy array to tensor and add channel dimension
            waveform = torch.FloatTensor(waveform).unsqueeze(0)  # [1, length]

            # Pad or crop to max_length
            waveform = self._process_length(waveform)

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zero waveform on error
            waveform = torch.zeros(1, self.max_length)

        return waveform, label, audio_id

    def _process_length(self, waveform):
        """
        Pad or crop waveform to max_length

        Args:
            waveform: [1, length] tensor

        Returns:
            waveform: [1, max_length] tensor
        """
        current_length = waveform.shape[1]

        if current_length > self.max_length:
            # Crop
            if self.random_start:
                start_idx = np.random.randint(0, current_length - self.max_length + 1)
                waveform = waveform[:, start_idx:start_idx + self.max_length]
            else:
                waveform = waveform[:, :self.max_length]

        elif current_length < self.max_length:
            # Pad
            if self.padding_type == 'repeat':
                # Repeat waveform until it reaches max_length
                num_repeats = (self.max_length // current_length) + 1
                waveform = waveform.repeat(1, num_repeats)
                waveform = waveform[:, :self.max_length]
            elif self.padding_type == 'zero':
                # Zero padding
                pad_length = self.max_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, pad_length), value=0)
            else:
                raise ValueError(f"Unknown padding_type: {self.padding_type}")

        return waveform


class NoiseAugmentedDataset(Dataset):
    """
    Dataset with noise-based domain grouping for multi-LoRA training

    8-domain grouping:
        g0: clean
        g1: background_music + background_noise
        g2: auto_tune
        g3: band_pass_filter
        g4: echo
        g5: pitch_shift + time_stretch
        g6: gaussian_noise
        g7: reverberation
    """
    DOMAIN_MAPPING = {
        'clean': 'g0',
        'background_music': 'g1',
        'background_noise': 'g1',
        'auto_tune': 'g2',
        'band_pass_filter': 'g3',
        'echo': 'g4',
        'pitch_shift': 'g5',
        'time_stretch': 'g5',
        'gaussian_noise': 'g6',
        'reverberation': 'g7',
    }

    def __init__(
        self,
        protocol_path,
        audio_dir,
        augmentation_type='clean',
        sample_rate=16000,
        max_length=64600,
        padding_type='repeat',
        random_start=False
    ):
        """
        Args:
            protocol_path: Path to ASVspoof protocol file
            audio_dir: Directory containing audio files
            augmentation_type: Noise augmentation type
            sample_rate: Target sample rate
            max_length: Maximum audio length in samples
            padding_type: Padding type ('repeat' or 'zero')
            random_start: Whether to randomly crop audio
        """
        self.base_dataset = ASVspoofDataset(
            protocol_path=protocol_path,
            audio_dir=audio_dir,
            augmentation_type=augmentation_type,
            sample_rate=sample_rate,
            max_length=max_length,
            padding_type=padding_type,
            random_start=random_start
        )

        self.augmentation_type = augmentation_type
        self.domain = self._get_domain()

        print(f"Domain: {self.domain} (augmentation: {augmentation_type})")

    def _get_domain(self):
        """Get domain group for current augmentation type"""
        return self.DOMAIN_MAPPING.get(self.augmentation_type, 'g0')

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Returns:
            waveform: [1, max_length] tensor
            label: 0 (bonafide) or 1 (spoof)
            audio_id: audio file ID
            domain: domain group (g0-g7)
        """
        waveform, label, audio_id = self.base_dataset[idx]
        return waveform, label, audio_id, self.domain


def collate_fn(batch):
    """
    Collate function for DataLoader

    Args:
        batch: List of (waveform, label, audio_id) tuples

    Returns:
        waveforms: [batch, max_length] tensor (channel dimension removed for AASIST/Conformer)
        labels: [batch] tensor
        audio_ids: list of audio IDs
    """
    waveforms, labels, audio_ids = zip(*batch)

    waveforms = torch.stack(waveforms, dim=0)  # [batch, 1, max_length]
    waveforms = waveforms.squeeze(1)  # [batch, max_length] - remove channel dimension
    labels = torch.tensor(labels, dtype=torch.long)  # [batch]

    return waveforms, labels, list(audio_ids)


def collate_fn_with_domain(batch):
    """
    Collate function for NoiseAugmentedDataset with domain information

    Args:
        batch: List of (waveform, label, audio_id, domain) tuples

    Returns:
        waveforms: [batch, 1, max_length] tensor
        labels: [batch] tensor
        audio_ids: list of audio IDs
        domains: list of domain groups
    """
    waveforms, labels, audio_ids, domains = zip(*batch)

    waveforms = torch.stack(waveforms, dim=0)  # [batch, 1, max_length]
    labels = torch.tensor(labels, dtype=torch.long)  # [batch]

    return waveforms, labels, list(audio_ids), list(domains)
