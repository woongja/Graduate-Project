import os
import pandas as pd
from tqdm import tqdm
import yaml
from augmentation import AutoTuneAugmentor

ROOT_DIR = '/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/datasets'
PROTOCOL_CSV = '/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/meta_file/meta_clean.csv'
OUTPUT_CSV = 'meta_file_autotune.csv'

with open("/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/augmentation_config.yaml", "r") as f:
    AUG_CONFIG = yaml.safe_load(f)
BASE_AUTO_TUNE_CONFIG = AUG_CONFIG['auto_tune']

def main():
    df = pd.read_csv(PROTOCOL_CSV)
    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        clean_path = row['File_path']
        group = row['Group']
        label2 = row['Label2']
        top_folder = 'bonafide' if label2 == 'bonafide' else 'spoof'
        fname = os.path.basename(clean_path).replace('.wav', '')

        out_dir = os.path.join(ROOT_DIR, top_folder, 'noise', 'auto_tune')
        os.makedirs(out_dir, exist_ok=True)
        out_wav_path = os.path.join(out_dir, f'{fname}_auto_tune.wav')

        # AutoTune config 복사 + output_path, out_format 추가
        config = dict(BASE_AUTO_TUNE_CONFIG)
        config['output_path'] = out_wav_path
        config['out_format'] = 'wav'

        augmentor = AutoTuneAugmentor(config)
        augmentor.load(clean_path)
        augmentor.transform()
        augmentor.augmented_audio.export(out_wav_path, format="wav")

        records.append({
            'file_path': out_wav_path,
            'group': group,
            'label2': label2,
            'label1': 'auto_tune',
            'ratio': augmentor.ratio
        })

    pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)

if __name__ == '__main__':
    main()
