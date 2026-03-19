import os
from augmentation import (
    BackgroundNoiseAugmentorDeepen as BackgroundNoiseAugmentor,
    BackgroundMusicAugmentorDeepen as BackgroundMusicAugmentor,
    GaussianAugmentorV1,
    HighPassFilterAugmentor,
    LowPassFilterAugmentor,
    PitchAugmentor,
    TimeStretchAugmentor,
    AutoTuneAugmentor,
    EchoAugmentorDeepen as EchoAugmentor,
    ReverbAugmentor,
)
import yaml

# 테스트할 clean 파일 경로와 config yaml 파일 경로
CLEAN_PATH = '/home/woongjae/noise-tracing/new_dataset/Dataset/Bonafide/Clean/14-208-0000.wav'
with open('/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/augmentation_config.yaml', 'r') as f:
    AUG_CONFIG = yaml.safe_load(f)

AUGMENTATION_CLASSES = {
    'background_noise': BackgroundNoiseAugmentor,
    'background_music': BackgroundMusicAugmentor,
    'gaussian_noise': GaussianAugmentorV1,
    'high_pass_filter': HighPassFilterAugmentor,
    'low_pass_filter': LowPassFilterAugmentor,
    'pitch_shift': PitchAugmentor,
    'time_stretch': TimeStretchAugmentor,
    # 'auto_tune': AutoTuneAugmentor,
    'echo': EchoAugmentor,
    'reverberation': ReverbAugmentor,
}

# 저장 폴더
SAVE_DIR = "/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/aug_test"
os.makedirs(SAVE_DIR, exist_ok=True)

for aug_name, aug_class in AUGMENTATION_CLASSES.items():
    config = dict(AUG_CONFIG[aug_name])  # config 복사
    config["output_path"] = SAVE_DIR
    config["out_format"] = "wav"
    augmentor = aug_class(config)
    augmentor.load(CLEAN_PATH)
    augmentor.transform()
    # 파일명 예시: cleanfile_background_noise.wav
    save_name = f"{os.path.basename(CLEAN_PATH).replace('.wav','')}_{aug_name}.wav"
    out_path = os.path.join(SAVE_DIR, save_name)
    augmentor.augmented_audio.export(out_path, format="wav")
    print(f"{aug_name} 샘플 저장: {out_path}, ratio: {getattr(augmentor, 'ratio', None)}")
