# Noise Classification Development Log

> **프로젝트**: Stage 1 - 노이즈 타입 분류 모델 개발
> **최종 업데이트**: 2025-03-25

---

## 📑 목차

1. [모델 아키텍처 현황](#모델-아키텍처-현황)
2. [개발 이력 (Timeline)](#개발-이력-timeline)
3. [코드 구조](#코드-구조)
4. [실험 설정 및 결과](#실험-설정-및-결과)
5. [TODO 리스트](#todo-리스트)
6. [기술 노트](#기술-노트)
7. [문제 해결 기록](#문제-해결-기록)

---

## 모델 아키텍처 현황

### 구현된 모델 (11개)

#### 1. Transformer 기반 모델

**AST (Audio Spectrogram Transformer)**
- 파일: `model/ast_model.py`
- Config: `config/ast.yaml`
- 특징:
  - ImageNet pretrained ViT 활용
  - Mel-spectrogram → patch embedding
  - Input: [B, 1, 128, target_length]
  - fstride=10, tstride=10

**SSAST (Self-Supervised Audio Spectrogram Transformer)**
- 파일: `model/ssast_model.py`
- Config:
  - `config/ssast_tiny_patch_400.yaml`
  - `config/ssast_small_patch_400.yaml`
  - `config/ssast_base_patch_400.yaml`
- 특징:
  - Self-supervised pretrained
  - Patch size: 16x16 (fshape, tshape)
  - 3가지 크기 variants

**HTSAT (Hierarchical Token-Semantic Audio Transformer)**
- 파일: `model/htsat_model.py`
- Config: `config/htsat.yaml`
- 특징:
  - Swin Transformer 기반
  - Hierarchical attention mechanism
  - Pretrained path 옵션 지원

**DASS (DeiT-based Audio Spectrogram Spectrogram)**
- 파일: `model/dass_model.py`
- 특징:
  - DeiT (Data-efficient image Transformers) 기반
  - Distillation 기법 활용

#### 2. Self-Supervised Learning 기반 모델

**HuBERT**
- 파일: `model/hubert_model.py`
- Config: `config/hubert.yaml`
- 특징:
  - Waveform 직접 처리 (no spectrogram)
  - Pretrained: `facebook/hubert-base-audioset`
  - Pooling: mean/first/attention 선택 가능
  - Feature extractor freeze 옵션

**Wav2Vec2**
- 파일: `model/wav2vec2_model.py`
- Config: `config/wav2vec2.yaml`
- 특징:
  - Waveform 직접 처리
  - Pretrained: `facebook/wav2vec2-base-audioset`
  - Pooling 방식 동일 (mean/first/attention)

**CLAP (Contrastive Language-Audio Pretraining)**
- 파일: `model/clap_model.py`
- 특징:
  - Audio-text contrastive learning
  - HTSAT audio encoder 사용
  - Text encoder frozen

#### 3. CNN+RNN 기반 모델

**CNN8RNN**
- 파일: `model/cnn8rnn_model.py`
- Config: `config/cnn8rnn.yaml`
- 특징:
  - PANN (Pretrained Audio Neural Networks) 기반
  - 8-layer CNN + GRU
  - Pretrained on AudioSet

**CNNLSTM (Baseline)**
- 파일: `model/cnnlstm.py`
- Config: `config/cnnlstm.yaml`
- 특징:
  - Lightweight baseline
  - CNN feature extraction + BiLSTM
  - Input: Mel-spectrogram only

**CNNLSTM_2FF (2-Feature Fusion)**
- 파일: `model/cnnlstm_2ff.py`
- Config: `config/cnnlstm_2ff.yaml`
- 특징:
  - **Spectrogram + F0** fusion
  - F0 extraction: CREPE or YIN
  - Separate CNN+LSTM branches
  - Late fusion (concatenation)

**CNNLSTM_3FF (3-Feature Fusion)**
- 파일: `model/cnnlstm_3ff.py`
- Config: `config/cnnlstm_3ff.yaml`
- 특징:
  - **Spectrogram + MFCC + F0** fusion
  - 3개 독립 branch (각 CNN+LSTM)
  - Feature dimensions: [512, 512, 512] → 1536
  - Classifier: 1536 → 512 → 10

**CNNLSTM_3FF_Interaction** ⭐ NEW (2025-03-25)
- 파일: `model/cnnlstm_3ff_interaction.py`
- Config: `config/cnnlstm_3ff_interaction.yaml`
- 특징:
  - 3FF 기반 + **Deep Interaction Layer**
  - Interaction: 1536 → 1024 → 1536
  - GELU 활성화 함수
  - Classifier: 1536 → 1024 → 512 → 10 (3-layer)
  - 목적: Feature interaction 강화

**CNNLSTM_3FF_Weight** ⭐ NEW (2025-03-25)
- 파일: `model/cnnlstm_3ff_weight.py`
- Config: `config/cnnlstm_3ff_weight.yaml`
- 특징:
  - 3FF 기반 + **Feature-Level Attention**
  - Attention module: 1536 → 256 → 3 (Sigmoid)
  - Adaptive weight learning: [w_spec, w_mfcc, w_f0]
  - Weighted fusion: Σ(w_i × feature_i)
  - GELU 활성화 함수
  - Classifier: 1536 → 1024 → 512 → 10
  - 목적: 각 feature의 중요도 동적 학습

### 학습 완료된 모델 (10개)

| 모델 | 체크포인트 경로 | 크기 | 파라미터 수 (예상) | 학습 완료일 |
|------|----------------|------|------------------|------------|
| cnn8rnn | `out/cnn8rnn/best.pth` | 24 MB | ~6M | 2025-03-2X |
| cnnlstm | `out/cnnlstm/best.pth` | 5.2 MB | ~1.3M | 2025-03-2X |
| cnnlstm_2ff | `out/cnnlstm_2ff/best.pth` | 8.7 MB | ~2.2M | 2025-03-2X |
| cnnlstm_3ff | `out/cnnlstm_3ff/best.pth` | 12 MB | ~3M | 2025-03-2X |
| htsat | `out/htsat/best.pth` | 112 MB | ~28M | 2025-03-2X |
| hubert | `out/hubert/best.pth` | 361 MB | ~90M | 2025-03-2X |
| ssast_tiny | `out/ssast_tiny_patch_400/best.pth` | 23 MB | ~5.8M | 2025-03-2X |
| ssast_small | `out/ssast_small_patch_400/best.pth` | 86 MB | ~21.5M | 2025-03-2X |
| ssast_base | `out/ssast_base_patch_400/best.pth` | 333 MB | ~83M | 2025-03-2X |
| wav2vec2 | `out/wav2vec2/best.pth` | 361 MB | ~90M | 2025-03-2X |

### 학습 진행 중/대기 중

| 모델 | 상태 | 비고 |
|------|------|------|
| cnnlstm_3ff_interaction | ✅ 완료 | 29MB checkpoint |
| cnnlstm_3ff_weight | ⏳ 재학습 대기 | I/O 에러 해결 완료 |

---

## 개발 이력 (Timeline)

### 2025-03-25: CNN-LSTM 3FF 변형 모델 추가

#### 구현한 모델
1. **cnnlstm_3ff_interaction** - Interaction Layer 추가
   - 동기: Feature fusion 후 feature 간 상호작용 강화
   - 구조:
     ```
     Spec Branch (512) ─┐
     MFCC Branch (512) ─┤→ Concat (1536) → Interaction (1024) → (1536) → Classifier
     F0 Branch (512)   ─┘
     ```
   - Interaction Layer:
     - Linear(1536, 1024) + BatchNorm + GELU + Dropout
     - Linear(1024, 1536) + BatchNorm + GELU
   - Classifier (3-layer):
     - 1536 → 1024 (GELU + Dropout)
     - 1024 → 512 (GELU + Dropout)
     - 512 → 10

2. **cnnlstm_3ff_weight** - Feature-Level Attention 추가
   - 동기: 각 feature (spec, mfcc, f0)의 중요도를 동적으로 학습
   - 구조:
     ```
     Spec (512) ─┐
     MFCC (512) ─┤→ Concat (1536) → Attention Net → [w_spec, w_mfcc, w_f0]
     F0 (512)   ─┘                                          ↓
                                                    Weighted Fusion
                                                          ↓
                                                     Classifier
     ```
   - Attention Mechanism:
     ```python
     class FeatureAttention:
         self.attention = nn.Sequential(
             nn.Linear(1536, 256),
             nn.GELU(),
             nn.Linear(256, 3),
             nn.Sigmoid()  # [w_spec, w_mfcc, w_f0]
         )

         weighted_spec = spec_feat * w_spec
         weighted_mfcc = mfcc_feat * w_mfcc
         weighted_f0 = f0_feat * w_f0
         fused = concat([weighted_spec, weighted_mfcc, weighted_f0])
     ```

#### train.py 수정 사항
- `build_model()` (line 122-136): 두 모델 추가
  ```python
  elif args.model == 'cnnlstm_3ff_interaction':
      from model.cnnlstm_3ff_interaction import create_cnnlstm_3ff_interaction
      model = create_cnnlstm_3ff_interaction(...)
  elif args.model == 'cnnlstm_3ff_weight':
      from model.cnnlstm_3ff_weight import create_cnnlstm_3ff_weight
      model = create_cnnlstm_3ff_weight(...)
  ```

- `build_loaders()` (line 199): 3FF 계열 모델 통합 처리
  ```python
  elif args.model in ('cnnlstm_3ff', 'cnnlstm_3ff_interaction', 'cnnlstm_3ff_weight'):
      from datautils.dataset_cnnlstm_3ff import CNNLSTM_3FF_Dataset, collate_3ff
      ...
  ```

- `build_eval_loader()` (line 291): Eval 로더 통합
- `_forward()` (line 366): Forward pass 통합
- `collate_fn` (line 251): collate_3ff 적용

#### Config 파일 생성
**config/cnnlstm_3ff_interaction.yaml**
```yaml
model: cnnlstm_3ff_interaction
branch_output_dim: 512
interaction_hidden_dim: 1024
dropout: 0.5
f0_method: crepe
clip_duration: 10.0
batch_size: 12
learning_rate: 1.0e-4
```

**config/cnnlstm_3ff_weight.yaml**
```yaml
model: cnnlstm_3ff_weight
branch_output_dim: 512
dropout: 0.5
f0_method: crepe
clip_duration: 10.0
batch_size: 12
learning_rate: 1.0e-4
```

#### 모델 저장 I/O 에러 해결
- **문제**: `RuntimeError: basic_ios::clear: iostream error`
  - 파일: `cnnlstm_3ff_weight/best.pth` (0 byte 손상)
  - 원인: 파일 시스템 I/O 버퍼 플러시 실패
  - 에러 위치: `torch.save()` 시 8.4MB 쓰기 중단

- **해결**:
  ```bash
  rm -f out/cnnlstm_3ff_weight/best.pth
  rm -f out/cnnlstm_3ff_weight/train_state.npz
  ```
  - 재학습 준비 완료 (디스크 공간 충분: 735GB)

### 2025-03-20 ~ 2025-03-24: 기본 모델 학습

- ✅ 10개 모델 학습 완료
- ✅ TensorBoard 로깅 설정
- ✅ Early stopping, n-best checkpoint 저장
- ✅ Resume 기능 구현
- ✅ 평가 스크립트 구현

---

## 코드 구조

```
01_Noise_Classification/
├── model/                          # 모델 정의
│   ├── ast_model.py
│   ├── ssast_model.py
│   ├── htsat_model.py
│   ├── hubert_model.py
│   ├── wav2vec2_model.py
│   ├── cnn8rnn_model.py
│   ├── cnnlstm.py
│   ├── cnnlstm_2ff.py
│   ├── cnnlstm_3ff.py
│   ├── cnnlstm_3ff_interaction.py  ⭐ NEW
│   ├── cnnlstm_3ff_weight.py       ⭐ NEW
│   └── ...
│
├── config/                         # YAML 설정 파일
│   ├── ast.yaml
│   ├── ssast_{tiny,small,base}_patch_400.yaml
│   ├── htsat.yaml
│   ├── hubert.yaml
│   ├── wav2vec2.yaml
│   ├── cnn8rnn.yaml
│   ├── cnnlstm.yaml
│   ├── cnnlstm_2ff.yaml
│   ├── cnnlstm_3ff.yaml
│   ├── cnnlstm_3ff_interaction.yaml  ⭐ NEW
│   └── cnnlstm_3ff_weight.yaml       ⭐ NEW
│
├── datautils/                      # 데이터 로더
│   ├── dataset_ast.py
│   ├── dataset_htsat.py
│   ├── dataset_hubert.py
│   ├── dataset_cnnlstm.py
│   ├── dataset_cnnlstm_2ff.py
│   └── dataset_cnnlstm_3ff.py      # 2FF, 3FF, 3FF variants 공용
│
├── scripts/                        # 학습 스크립트
│   └── train.sh
│
├── train.py                        # 통합 학습 엔트리포인트
├── out/                            # 체크포인트 저장
│   ├── cnnlstm_3ff/
│   ├── cnnlstm_3ff_interaction/
│   ├── cnnlstm_3ff_weight/
│   └── ...
│
├── runs/                           # TensorBoard 로그
│   ├── cnnlstm_3ff/
│   ├── cnnlstm_3ff_interaction/
│   └── ...
│
├── protocols/                      # 데이터 프로토콜
│   ├── train_protocol.txt
│   └── eval_protocol.txt
│
├── pretrained_model/               # Pretrained weights
│   ├── cnn8rnn-audioset-sed/
│   ├── hubert-base-audioset/
│   └── wav2vec2-base-audioset/
│
└── AudioMamba/                     # (불완전 - 통합 보류)
```

---

## 실험 설정 및 결과

### 공통 설정

**데이터 전처리**
- Sample rate: 16kHz
- Duration: 10초 (고정 길이)
- Padding: repeat
- 정규화: model-specific (AST: -4.27/4.57, HTSAT: per-sample)

**학습 설정**
- Optimizer: Adam
- Learning rate: 1e-4 ~ 1e-5 (model-dependent)
- Batch size: 12~32 (model-dependent)
- Early stopping patience: 5~10 epochs
- Loss: CrossEntropyLoss
- Metric: Accuracy (top-1)

**Data Augmentation**
- SpecAugment (AST, SSAST, HTSAT):
  - Frequency masking: 48 bins
  - Time masking: 192 frames

### 모델별 하이퍼파라미터

| 모델 | LR | Batch Size | Epochs | Special Settings |
|------|-----|-----------|--------|------------------|
| AST | 1e-5 | 32 | 30 | ImageNet pretrain |
| SSAST (base) | 1e-5 | 32 | 30 | SSAST pretrain |
| HTSAT | 1e-5 | 32 | 30 | AudioSet pretrain |
| HuBERT | 1e-5 | 32 | 30 | Freeze feature extractor |
| Wav2Vec2 | 1e-5 | 32 | 30 | Freeze feature extractor |
| CNN8RNN | 1e-4 | 32 | 30 | AudioSet pretrain |
| CNNLSTM | 1e-4 | 32 | 30 | From scratch |
| CNNLSTM_2FF | 1e-4 | 12 | 100 | F0: CREPE |
| CNNLSTM_3FF | 1e-4 | 12 | 100 | F0: CREPE |
| CNNLSTM_3FF_Interaction | 1e-4 | 12 | 100 | F0: CREPE, GELU |
| CNNLSTM_3FF_Weight | 1e-4 | 12 | 100 | F0: CREPE, GELU |

### 성능 비교표 (추후 업데이트)

| 모델 | Dev Acc | Eval Acc | 파라미터 수 | 추론 시간 | 비고 |
|------|---------|----------|------------|----------|------|
| ssast_base | TBD | TBD | ~83M | TBD | Pretrained |
| htsat | TBD | TBD | ~28M | TBD | Pretrained |
| hubert | TBD | TBD | ~90M | TBD | SSL + Freeze |
| wav2vec2 | TBD | TBD | ~90M | TBD | SSL + Freeze |
| cnn8rnn | TBD | TBD | ~6M | TBD | Pretrained |
| cnnlstm_3ff_weight | TBD | TBD | ~3M | TBD | Attention |
| cnnlstm_3ff_interaction | TBD | TBD | ~3M | TBD | Interaction |
| cnnlstm_3ff | TBD | TBD | ~3M | TBD | Baseline 3FF |
| cnnlstm_2ff | TBD | TBD | ~2.2M | TBD | 2-feature |
| cnnlstm | TBD | TBD | ~1.3M | TBD | Lightweight |

---

## TODO 리스트

### 긴급 (High Priority)
- [ ] **cnnlstm_3ff_weight 재학습 실행**
  ```bash
  cd scripts
  bash train.sh cnnlstm_3ff_weight
  ```
- [ ] **전체 모델 평가 실행**
  ```bash
  for model in $(ls out/*/best.pth); do
      python train.py --config config/${model}.yaml --is_eval \
          --model_path $model \
          --save_results results/${model}_eval.txt
  done
  ```

### 중요 (Medium Priority)
- [ ] **성능 비교표 작성**
  - Dev/Eval accuracy 수집
  - Confusion matrix 생성
  - 모델별 장단점 분석

- [ ] **최종 모델 선정**
  - 성능/효율성 trade-off 고려
  - Stage 2로 전달할 Noise Classifier 결정

- [ ] **Stage 2 데이터 준비**
  - 10-class → 8-domain 매핑 스크립트 작성
  - LoRA 학습용 프로토콜 파일 생성

### 낮음 (Low Priority)
- [ ] **AudioMamba 통합 재검토**
  - 저자 리포지토리 업데이트 확인
  - 완전한 코드 공개 시 통합 시도

- [ ] **모델 앙상블 실험**
  - Top-3 모델 앙상블
  - Soft voting vs Hard voting

- [ ] **Hyperparameter tuning**
  - Learning rate sweep
  - Batch size optimization

---

## 기술 노트

### Feature Fusion 방법론

#### 1. Early Fusion (사용 안 함)
- Feature 추출 전 결합
- 예: Raw waveform concatenation
- 단점: 입력 차원 폭발

#### 2. Late Fusion (CNNLSTM_3FF)
- 각 branch 독립적으로 feature 추출
- 최종 단계에서 concatenation
- 장점: 각 feature 특성 보존
- 단점: Feature 간 interaction 부족

#### 3. Interaction Layer (CNNLSTM_3FF_Interaction)
- Late fusion + Deep interaction network
- Feature 간 non-linear transformation
- 수식:
  ```
  z = concat([f_spec, f_mfcc, f_f0])  # 1536-dim
  z_int = W2(GELU(W1(z)))              # 1536 → 1024 → 1536
  y = Classifier(z_int)
  ```
- 장점: Feature 상호작용 학습
- 단점: 파라미터 증가

#### 4. Attention-based Fusion (CNNLSTM_3FF_Weight)
- Adaptive weighted fusion
- 각 feature의 중요도 동적 학습
- 수식:
  ```
  [w_s, w_m, w_f] = Sigmoid(Attention(concat([f_spec, f_mfcc, f_f0])))
  z = concat([w_s * f_spec, w_m * f_mfcc, w_f * f_f0])
  y = Classifier(z)
  ```
- 장점: 해석 가능성 (attention weights 시각화)
- 단점: 학습 불안정 가능성

### GELU vs ReLU

**GELU (Gaussian Error Linear Unit)**
```python
GELU(x) = x * Φ(x)  # Φ: standard Gaussian CDF
```
- 장점:
  - Smooth, differentiable
  - Transformer에서 ReLU보다 우수
  - Gradient flow 개선
- 적용: 3ff_interaction, 3ff_weight

**ReLU**
```python
ReLU(x) = max(0, x)
```
- 장점: 단순, 빠름
- 단점: Dying ReLU 문제
- 적용: 기존 CNNLSTM 계열

### F0 추출 방법

**CREPE (현재 사용)**
- Deep learning 기반
- 장점: 정확도 높음
- 단점: 느림 (~0.5초/10초 audio)

**YIN (대안)**
- Algorithm 기반
- 장점: 빠름
- 단점: 정확도 낮음

현재 선택: CREPE (정확도 우선)

---

## 문제 해결 기록

### 1. AudioMamba 통합 실패 (2025-03-25)

**문제**:
- GitHub 리포지토리 불완전
- `model/swin.py`, `sed_model.py` 누락
- `__init__.py`에서 import 에러

**조사**:
```bash
cd AudioMamba
git ls-tree -r HEAD | wc -l  # 15 files only
```
- 실제 모델 코드 없음
- Config 파일과 README만 존재

**결정**:
- AudioMamba 통합 보류
- 현재 11개 모델로 충분

**향후 계획**:
- 저자가 완전한 코드 공개 시 재검토
- 또는 Vision Mamba 기반 직접 구현

### 2. 모델 저장 I/O 에러 (2025-03-25)

**증상**:
```
RuntimeError: basic_ios::clear: iostream error
RuntimeError: [enforce fail at inline_container.cc:668] .
unexpected pos 8392704 vs 8392600
```

**원인**:
- 파일 시스템 버퍼 플러시 실패
- PyTorch 예상 크기 vs 실제 쓰기 크기 불일치 (104 bytes)

**영향**:
- `cnnlstm_3ff_weight/best.pth`: 0 byte (손상)

**해결**:
```bash
rm -f out/cnnlstm_3ff_weight/best.pth
rm -f out/cnnlstm_3ff_weight/train_state.npz
```
- 디스크 공간 확인: 735GB 여유 (충분)
- 재학습 실행 대기

**예방**:
- 현재는 간단한 재학습으로 해결
- 향후 필요 시 atomic save 구현 고려:
  ```python
  temp_path = save_path + '.tmp'
  torch.save(model.state_dict(), temp_path)
  os.replace(temp_path, save_path)
  ```

### 3. F0 추출 속도 이슈 (해결)

**문제**:
- CREPE F0 추출 느림 (~0.5초/sample)
- DataLoader bottleneck

**해결**:
- `num_workers=2` (더 늘리면 CUDA OOM)
- `batch_size=12` (작게 유지)
- 사전 캐싱 미사용 (디스크 용량 부족)

---

## 참고 자료

### 논문 참고
- AST: Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021
- SSAST: Gong et al., "SSAST: Self-Supervised Audio Spectrogram Transformer", AAAI 2022
- HT-SAT: Chen et al., "HTS-AT: A Hierarchical Token-Semantic Audio Transformer", ICASSP 2022
- HuBERT: Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning", IEEE/ACM TASLP 2021
- Wav2Vec2: Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning", NeurIPS 2020

### 관련 프로젝트
- Stage 2 (LoRA Training): `../02_LoRA_Training/`
- Stage 3 (DNA-MultiLoRA): `../03_DNA_MultiLoRA_Framework/`

---

**Last Updated**: 2025-03-25 17:20 KST
