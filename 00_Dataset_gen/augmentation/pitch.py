from .base import BaseAugmentor
from .utils import librosa_to_pydub
import random
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from audiomentations import PitchShift
import logging
logger = logging.getLogger(__name__)

class PitchAugmentor(BaseAugmentor):
    """
    Pitch augmentation
    Generates versions of audio pitch-shifted from -5 to +5 semitones
    """
    def __init__(self, config: dict):
        """
        This method initializes the `PitchAugmentor` object.
        min_abs_semitones: float • minimum absolute semitone shift
        max_abs_semitones: float • maximum absolute semitone shift
        Sign (positive/negative) is chosen randomly, so the effective range is
        [-max_abs, -min_abs] ∪ [+min_abs, +max_abs].
        :param config: dict, configuration dictionary
        """
        super().__init__(config)
        self.min_abs_semitones = config["min_abs_semitones"]
        self.max_abs_semitones = config["max_abs_semitones"]

    def transform(self):
        """
        Transform the audio by pitch shifting.
        Randomly selects an absolute shift in [min_abs, max_abs] and a random sign.
        """
        abs_steps = np.random.uniform(self.min_abs_semitones, self.max_abs_semitones)
        sign = np.random.choice([-1, 1])
        n_steps = sign * abs_steps
        # librosa로 적용
        augmented_data = librosa.effects.pitch_shift(y = self.data, sr = self.sr, n_steps=n_steps)
        # Transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)
        self.ratio = f"pitch:{n_steps}"
 