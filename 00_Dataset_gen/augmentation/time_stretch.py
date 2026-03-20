from .base import BaseAugmentor
from audiomentations import TimeStretch
import logging
from .utils import librosa_to_pydub
logger = logging.getLogger(__name__)
import random
import librosa
import numpy as np

class TimeStretchAugmentor(BaseAugmentor):
    """
    Time stretch augmentation
    Modifies audio speed by a factor between 0.8 (slower) and 1.2 (faster)
    using librosa.effects.time_stretch()
    """
    def __init__(self, config: dict = None):
        """
        Initialize the TimeStretchAugmentor
        
        :param config: dict, optional configuration dictionary
                      If None, default range of 0.8 to 1.2 will be used
        """
        super().__init__(config or {})
        
        # Set default values based on the description
        self.min_rate = config["min_factor"]
        self.max_rate = config["max_factor"]
        self.time_stretch_transform = TimeStretch(
            min_rate=self.min_rate,
            max_rate=self.max_rate,
            method="librosa_phase_vocoder",
            leave_length_unchanged=True,
            p=1.0
        )
    
    def transform(self, rate=None):
        """
        Time stretch the audio using librosa.effects.time_stretch()
        
        :param rate: Optional specific stretch rate to apply
                    If None, applies a random rate in the specified range
        :return: The time-stretched audio segment
        """
        rate = np.random.uniform(self.min_rate, self.max_rate)
        ts = TimeStretch(min_rate=rate, max_rate=rate, p=1.0)
        augmented_data = ts(self.data, sample_rate=self.sr)
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)
        self.ratio = f"stretch:{rate:.2f}"
    
    