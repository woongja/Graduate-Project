# Graduate Project — Audio Deepfake Detection via Noise-Aware Multi-LoRA Framework

> 졸업 논문 프로젝트: 노이즈 환경을 고려한 음성 위조 탐지 시스템 연구

---

## 📌 Overview

본 프로젝트는 실제 환경에서 발생하는 다양한 **노이즈 조건** 하에서도 강인한 **음성 위조(Audio Deepfake) 탐지**를 수행하는 시스템을 연구합니다.

음성 위조 탐지 모델은 깨끗한 음성(clean speech)에 대한 성능은 우수하지만, 배경 잡음·에코·피치 변조 등 실세계 음성 변형에는 취약합니다. 이 연구는 다음 핵심 질문에 답합니다:

> **"노이즈 종류를 먼저 분류하고, 그에 특화된 경량 어댑터(LoRA)를 선택적으로 적용하면 탐지 성능이 향상되는가?"**

이를 위해 아래 4단계 파이프라인을 구성합니다.

---

## 🗂️ Project Structure

```
Graduate-Project/
├── 00_Dataset_gen/          # 노이즈 증강 데이터셋 생성 파이프라인
├── 01_Noise_Classification/ # 노이즈 종류 분류 모델 학습 및 평가  ← 현재 진행 중
├── 02_LoRA_Training/        # 노이즈별 LoRA 어댑터 학습
└── 03_DNA_MultiLoRA_Framework/ # 통합 추론 프레임워크 (DNA-Multi-LoRA)
```

---

## 🔬 Research Pipeline

### Stage 0 — 데이터셋 생성 (`00_Dataset_gen`) ✅

원본 음성 데이터에 **10가지 노이즈 증강**을 적용하여 균등 분포의 학습 데이터셋을 구축합니다.

| 클래스 | 설명 |
|---|---|
| `clean` | 원본 (증강 없음) |
| `background_noise` | 배경 잡음 |
| `background_music` | 배경 음악 |
| `gaussian_noise` | 가우시안 노이즈 |
| `band_pass_filter` | 대역 통과 필터 (high + low pass) |
| `echo` | 에코 |
| `pitch_shift` | 피치 변조 |
| `time_stretch` | 속도 변조 |
| `reverberation` | 잔향 (RIR 기반) |
| `auto_tune` | 오토튠 효과 |

- **분할 전략**: speaker-disjoint, train / dev / eval = 5:2:3
- **출력**: `Datasets/noise_dataset/augmented/{split}/{aug_type}/`

---

### Stage 1 — 노이즈 분류기 (`01_Noise_Classification`) 🔄 *진행 중*

증강된 음성으로부터 **노이즈 종류를 10-class 분류**하는 모델을 학습합니다.  
이 분류기의 출력은 Stage 3에서 적절한 LoRA 어댑터를 선택하는 **라우팅 신호**로 사용됩니다.

#### 지원 모델

| 모델 | 유형 | 특징 |
|---|---|---|
| `AST` | Spectrogram Transformer | ImageNet/AudioSet 프리트레인 |
| `SSAST` | Self-supervised AST | 패치 기반 self-supervised |
| `HTSAT` | Swin Transformer | 계층적 오디오 Transformer |
| `HuBERT` | SSL Waveform | 파형 직접 처리 |
| `Wav2Vec2` | SSL Waveform | 파형 직접 처리 |
| `CLAP` | Contrastive (Audio+Text) | HTSAT 기반 인코더 |
| `DASS` | Spectrogram Transformer | DeiT 기반 |
| `CNN8RNN` | CNN + RNN | PANN 기반 경량 모델 |
| `CNNLSTM` | CNN + LSTM | 소형 경량 베이스라인 |

#### 실행 방법

```bash
cd 01_Noise_Classification

# 학습
python train.py --config config/ast.yaml --is_train

# 평가
python train.py --config config/ast.yaml --is_eval \
    --model_path out/ast/best.pth \
    --save_results out/eval_result.txt
```

#### 주요 설계

- **도메인 통일**: 모노 16kHz, 고정 길이 클리핑 / 패딩
- **SpecAugment**: 주파수·시간 마스킹으로 과적합 방지
- **학습 관리**: n-best 체크포인트 저장, early stopping, resume 지원
- **로깅**: TensorBoard (`runs/`) + 에폭별 loss/accuracy 출력
- **평가 출력**: `file_path / true_label / predicted_label / score / class_probs` TSV

---

### Stage 2 — LoRA 어댑터 학습 (`02_LoRA_Training`) 🔜 *예정*

각 노이즈 클래스에 특화된 **경량 LoRA 어댑터**를 음성 위조 탐지 베이스 모델에 학습합니다.  
노이즈 조건별로 독립적인 어댑터를 학습하여 최소한의 파라미터로 도메인 적응을 달성합니다.

---

### Stage 3 — DNA Multi-LoRA 추론 프레임워크 (`03_DNA_MultiLoRA_Framework`) 🔜 *예정*

Stage 1의 노이즈 분류기로 입력 음성의 노이즈 종류를 판단하고,  
해당 조건에 맞는 LoRA 어댑터를 **동적으로 선택(Dynamic Noise-Aware)**하여 탐지 성능을 극대화합니다.

---

## ⚙️ Environment

```bash
# Stage 0, 1
conda activate dataset       # 데이터 생성
conda activate <train_env>   # 모델 학습 (PyTorch + CUDA)

# autotune 증강 (별도 환경)
conda activate dataset_autotune
```

> Python 3.8+, PyTorch 2.x, CUDA 11.8+ 권장

---

## 📊 Evaluation Metric

노이즈 분류: **Top-1 Accuracy**, Confusion Matrix  
음성 위조 탐지 (Stage 3): **EER (Equal Error Rate)**, **t-DCF**
