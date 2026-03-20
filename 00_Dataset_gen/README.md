## 음성 증강 데이터셋 생성 파이프라인

음성 데이터에 다양한 노이즈를 적용해 증강 데이터셋을 생성하는 파이프라인입니다.
각 파일에 노이즈 1개를 무작위 배정하며, 전체 10개 클래스가 1:1 균등 비율이 되도록 구성됩니다.

---

## 증강 클래스

| 클래스 | 설명 |
|---|---|
| `background_noise` | 배경 노이즈 추가 |
| `background_music` | 배경 음악 추가 |
| `high_pass_filter` | 고역 통과 필터 (bandpass의 절반) |
| `low_pass_filter` | 저역 통과 필터 (bandpass의 절반) |
| `echo` | 에코 추가 |
| `pitch_shift` | 피치 변조 |
| `time_stretch` | 속도 변조 |
| `gaussian_noise` | 가우시안 노이즈 추가 |
| `reverberation` | 잔향 추가 |
| `clean` | 증강 없음 (원본 복사) |
| `auto_tune` | 오토튠 효과 (별도 환경에서 실행) |

`high_pass_filter`와 `low_pass_filter`는 `bandpass` 클래스를 반반 분담합니다.
`auto_tune`은 `dataset_autotune` conda 환경에서 별도로 실행됩니다.

---

## 환경 준비

```bash
# 일반 증강용
conda activate dataset
pip install -r requirements.txt

# autotune 증강용
conda activate dataset_autotune
pip install -r requirements_autotune.txt
```

`augmentation_config.yaml`에서 노이즈 소스 경로(`noise_path`, `music_path`, `rir_path`)를 실제 경로로 수정하세요.

---

## 실행 순서

### Step 1 — 메타 정규화
각 데이터셋의 원본 메타 파일을 통일된 포맷으로 변환합니다.

```bash
python3 scripts/normalize_meta.py
```

- 입력: `protocols/original_meta/*.csv`
- 출력: `protocols/processing_meta/*.csv` (columns: `file_path, speaker, utt, label1`)

### Step 2 — train / dev / eval 분할
speaker-disjoint 방식으로 5:2:3 비율 분할합니다.

```bash
bash scripts/00_data_split.sh
```

- 출력: `protocols/train.csv`, `protocols/dev.csv`, `protocols/eval.csv`

### Step 3 — 증강 데이터 생성
train / dev / eval 각 split에 대해 증강을 실행합니다.

```bash
bash scripts/01_data_generate.sh
```

- 출력: `Datasets/noise_dataset/augmented/{split}/{aug_type}/{speaker}_{utt}__{aug_type}.wav`
- 메타데이터: `Datasets/noise_dataset/augmented/metadata_{split}_{aug_type}.csv`

---

## 출력 구조

```
Datasets/noise_dataset/augmented/
  ├── train/
  │    ├── background_noise/
  │    ├── echo/
  │    ├── clean/
  │    └── auto_tune/ ...
  ├── dev/
  ├── eval/
  ├── metadata_train_background_noise.csv
  ├── metadata_train_echo.csv
  └── ...
```

메타데이터 CSV columns: `file_path, speaker, utt, dataset, label1, aug_type, split`

---

## 파일 구성

```
scripts/
  normalize_meta.py       # Step 1: 메타 정규화
  create_protocols.py     # Step 2: train/dev/eval 분할
  00_data_split.sh        # Step 2 실행 wrapper
  run_augmentations.py    # 증강 실행 코어 스크립트
  run_all_augmentations.sh
  01_data_generate.sh     # Step 3 실행 wrapper

augmentation_config.yaml  # 증강 파라미터 설정
protocols/
  original_meta/          # 원본 메타 파일 (수동 준비)
  processing_meta/        # normalize 후 통일 포맷
  train.csv / dev.csv / eval.csv
```
