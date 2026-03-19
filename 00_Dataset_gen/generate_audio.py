import os
import pandas as pd
from tqdm import tqdm
import yaml
import shutil
import random

# 증강 함수 모듈 import (여러 증강 클래스)
from augmentation import (
    BackgroundNoiseAugmentorDeepen as BackgroundNoiseAugmentor,
    BackgroundMusicAugmentorDeepen as BackgroundMusicAugmentor,
    GaussianAugmentorV1,
    HighPassFilterAugmentor,
    LowPassFilterAugmentor,
    FrequencyOperationAugmentorDeepen as FrequencyOperationAugmentor,
    PitchAugmentor,
    TimeStretchAugmentor,
    AutoTuneAugmentor,
    EchoAugmentorDeepen as EchoAugmentor,
    ReverbAugmentor,
)

AUGMENTATION_CLASSES = {
    'background_noise': BackgroundNoiseAugmentor,
    'background_music': BackgroundMusicAugmentor,
    'gaussian_noise': GaussianAugmentorV1,
    'high_pass_filter': HighPassFilterAugmentor,
    'low_pass_filter': LowPassFilterAugmentor,
    'pitch_shift': PitchAugmentor,
    'time_stretch': TimeStretchAugmentor,
 
    'echo': EchoAugmentor,
    'reverberation': ReverbAugmentor
}
AUGMENTATION_LIST = list(AUGMENTATION_CLASSES.keys())

ROOT_DIR = '/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/datasets'
PROTOCOL_CSV = '/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/meta_file/meta_clean.csv'   # clean 메타데이터 파일 경로
OUTPUT_CSV = 'meta_file.csv'  # 최종 결과 파일

with open("/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/augmentation_config.yaml", "r") as f:
    AUG_CONFIG = yaml.safe_load(f)

def main():
    df = pd.read_csv(PROTOCOL_CSV)
    records = []

    all_indices = list(df.index)
    random.shuffle(all_indices)
    n = len(all_indices)
    # bandpass 50:50 split
    high_pass_indices = set(all_indices[: n // 2])
    low_pass_indices = set(all_indices[n // 2 :])
    # manipulation 50:50 split (pitch/time_stretch)
    random.shuffle(all_indices)
    pitch_indices = set(all_indices[: n // 2])
    stretch_indices = set(all_indices[n // 2 :])
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 기본 정보 추출
        clean_path = row['File_path']
        group = row['Group']
        label2 = row['Label2']

        # clean 경로 -> bonafide/spoof 구분
        top_folder = 'bonafide' if label2 == 'bonafide' else 'spoof'
        fname = os.path.basename(clean_path).replace('.wav', '')

        # clean 파일에 대한 기록
        os.makedirs(os.path.join(ROOT_DIR, top_folder, 'clean'), exist_ok=True)
        out_clean_path = os.path.join(ROOT_DIR, top_folder, 'clean', os.path.basename(clean_path))
        records.append({
            'file_path': out_clean_path,
            'group': group,
            'label2': label2,
            'label1': 'clean',
            'ratio' : '-'
        })
        shutil.copy2(clean_path, out_clean_path)
        # 증강 종류별로 반복
        for aug_name in AUGMENTATION_LIST:
            aug_class = AUGMENTATION_CLASSES[aug_name]
            config = AUG_CONFIG.get(aug_name, {})  # 증강별 config 필요
            
            # 저장 경로 생성
            out_dir = os.path.join(ROOT_DIR, top_folder, 'noise', aug_name)
            os.makedirs(out_dir, exist_ok=True)
            out_wav_path = os.path.join(out_dir, f'{fname}_{aug_name}.wav')
            
            config['output_path'] = out_wav_path
            config['out_format'] = 'wav'
            
            # 인스턴스 생성, 오디오 로드
            augmentor = aug_class(config)
            augmentor.load(clean_path)
            augmentor.transform()
            augmentor.augmented_audio.export(out_wav_path, format="wav")
            # 메타데이터 기록
            records.append({
                'file_path': out_wav_path,
                'group': group,
                'label2': label2,
                'label1': aug_name,
                'ratio' : augmentor.ratio
            })
    # 결과 csv 저장
    pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)

if __name__ == '__main__':
    main()
