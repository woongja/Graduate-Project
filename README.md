# Graduate Project — Audio Deepfake Detection via Noise-Aware Multi-LoRA Framework

> 졸업 논문: 실환경 잡음에 강건성을 갖는 오디오 딥페이크 탐지 프레임워크 연구

---

## Overview

실환경 노이즈 조건에서 강인한 음성 위조 탐지 시스템 연구.

> **"노이즈 종류를 먼저 분류하고, 그에 특화된 경량 LoRA 어댑터를 동적으로 적용하면 탐지 성능이 향상되는가?"**

---

## Project Structure

```
Graduate-Project/
├── 00_Dataset_gen/              # 노이즈 증강 데이터셋 생성
├── 01_Noise_Classification/     # 노이즈 분류 모델 (5종)
├── 02_LoRA_Training/            # ADD 모델 LoRA 학습/평가/HP 탐색
└── 03_DNA_MultiLoRA_Framework/  # NC + LoRA 결합 파이프라인
```

---

## Research Pipeline

### Stage 0 — 데이터셋 생성 (`00_Dataset_gen`)

원본 음성에 10가지 노이즈 증강 적용.

| 노이즈 타입 | 도메인 |
|---|---|
| clean | D0 |
| background_noise | D1 |
| background_music | D2 |
| auto_tune | D3 |
| band_pass_filter (high+low) | D4 |
| echo | D5 |
| pitch_shift | D6 |
| time_stretch | D7 |
| gaussian_noise | D8 |
| reverberation | D9 |

---

### Stage 1 — 노이즈 분류기 (`01_Noise_Classification`)

10-class 노이즈 분류 모델 학습. Stage 3에서 LoRA 라우팅 신호로 사용.

| 모델 | 입력 |
|---|---|
| CNN8RNN-3FF-CrossModal | spec + mfcc + f0 |
| CNN8RNN-3FF | spec + mfcc + f0 |
| CNN+LSTM | mel-spectrogram |
| SSAST-tiny | fbank |
| HTS-AT | waveform (32kHz) |

```bash
cd 01_Noise_Classification
python train.py --config config/cnn8rnn_3ff_crossmodal.yaml --is_train
```

---

### Stage 2 — LoRA 학습 (`02_LoRA_Training`)

4종 ADD 모델에 도메인별 LoRA 학습. Optuna 기반 자동 HP 탐색 지원.

| ADD 모델 | Backbone | Pretrained |
|---|---|---|
| AASIST | XLSR + Graph ATT | `pretrained/aasist.pth` |
| ConformerTCM | XLSR + Conformer | `pretrained/conformertcm.pth` |
| XLSR-SLS | XLSR + Spectral Level Stats | `pretrained/XLSR-SLS.pth` |
| XLSR-Mamba | XLSR + Bidirectional Mamba | `pretrained/XLSR-Mamba-DF/model.safetensors` |

```bash
cd 02_LoRA_Training

# 평가
bash scripts/eval.sh aasist base asv19

# HP 탐색 (Optuna 자동)
python scripts/auto_lora/test.py --gpu MIG-xxx --model aasist
bash scripts/auto_lora/start.sh

# 도메인별 LoRA 학습
bash scripts/domain_lora/start.sh
```

---

### Stage 3 — DNA-MultiLoRA 프레임워크 (`03_DNA_MultiLoRA_Framework`)

NC 분류기로 노이즈 도메인 예측 → 해당 도메인 LoRA를 ADD 모델에 동적 적용 → 딥페이크 탐지.

```
Audio → [NC 분류기] → Domain ID (D0~D9)
                          ↓
       [ADD Model] + [Domain LoRA] → Spoof/Bonafide Score
```

```bash
cd 03_DNA_MultiLoRA_Framework

# 실험 생성 + 실행
python scripts/generate.py --clean
bash scripts/start.sh

# 대시보드
bash scripts/watch.sh
```

---

## Datasets

| 데이터셋 | 경로 |
|---|---|
| ASVspoof2019 eval | `/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_eval` |
| DF21 eval | `/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/DF21_eval/flac` |
| In-the-Wild | `/home/woongjae/ADD_LAB/Datasets/itw` |
| NC 데이터 | `/nvme3/Datasets/WJ/noise_dataset/NC_DF/` |

## Environment

```bash
conda activate sfm  # PyTorch + CUDA
```

## Evaluation Metric

- 노이즈 분류: Top-1 Accuracy
- 딥페이크 탐지: **EER (Equal Error Rate)**
