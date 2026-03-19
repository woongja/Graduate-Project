import librosa
import os
import typing
import soundfile as sf

import logging

logger = logging.getLogger(__name__)


class BaseAugmentor:
    """
    Basic augmentor class requires these config:
    aug_type: str, augmentation type
    output_path: str, output path
    out_format: str, output format
    """

    def __init__(self, config: dict):
        """
        This method initialize the `BaseAugmentor` object.
        """
        self.config = config
        self.aug_type = config["aug_type"]
        
        self.output_path = config["output_path"]
        self.out_format = config["out_format"]
        self.augmented_audio = None
        self.data = None
        self.sr = 16000

    def load(self, input_path: str):
        """
        Load audio file and normalize the data
        Use soundfile(sf) first, fallback to librosa if needed.
        self.data: audio data in numpy array
        :param input_path: path to the input audio file
        """
        self.input_path = input_path
        self.file_name = self.input_path.split("/")[-1].split(".")[0]

        try:
            # 1) soundfile로 읽기
            data, sr0 = sf.read(self.input_path, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"[WARN] soundfile failed for {self.input_path}, fallback to librosa: {e}")
            data, sr0 = librosa.load(self.input_path, sr=None, mono=False, dtype="float32")

        # 스테레오면 모노 다운믹스
        if data.ndim == 2:
            data = data.mean(axis=1)

        target_sr = self.sr or sr0
        if sr0 != target_sr:
            # librosa resample 사용
            data = librosa.resample(data, orig_sr=sr0, target_sr=target_sr, res_type="kaiser_best")
            sr0 = target_sr

        # 최종 저장
        self.data = data
        self.sr = sr0

    def transform(self):
        """
        Transform audio data (librosa load) to augmented audio data (pydub audio segment)
        Note that self.augmented_audio is pydub audio segment
        """
        raise NotImplementedError

    def save(self):
        """
        Save augmented audio data (pydub audio segment) to file
        self.out_format: output format
        This done the codec transform by pydub
        """
        self.augmented_audio.export(
            os.path.join(self.output_path, self.file_name + "." + self.out_format),
            format=self.out_format,
        )

