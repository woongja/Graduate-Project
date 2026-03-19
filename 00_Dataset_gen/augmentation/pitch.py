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
        min_semitones: float • unit: semitones • range: [-24.0, 24.0]
        max_semitones: float • unit: semitones • range: [-24.0, 24.0]
        :param config: dict, configuration dictionary
        """
        super().__init__(config)
        # Define the range of semitones (-5 to +5)
        self.min_semitones = config["min_semitones"]
        self.max_semitones = config["max_semitones"]
        
        # self.pitch_augmentor = PitchShift(
        #     min_semitones=self.min_semitones,
        #     max_semitones=self.max_semitones,
        #     method="librosa_phase_vocoder",
        #     p=1.0
        # )

    def transform(self):
        """
        Transform the audio by pitch shifting
        
        :param n_steps: Optional specific semitone shift to apply (-5 to +5)
                       If None, applies a random shift in the range
        :return: The pitch-shifted audio segment
        """
        n_steps = np.random.uniform(self.min_semitones, self.max_semitones)
        while n_steps == 0:
            n_steps = np.random.randint(self.min_semitones, self.max_semitones + 1)
        # librosa로 적용
        augmented_data = librosa.effects.pitch_shift(y = self.data, sr = self.sr, n_steps=n_steps)
        # Transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)
        self.ratio = f"pitch:{n_steps}"
 