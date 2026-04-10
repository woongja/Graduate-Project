# Graduate Project — Progress Tracker

> **프로젝트**: 노이즈 환경을 고려한 음성 위조 탐지 시스템 (Noise-Aware Multi-LoRA Framework)
> **최종 업데이트**: 2025-03-25

---

## 📍 현재 상태: Stage 1 진행 중 (Noise Classification)

### 전체 파이프라인 진행 상황

```
Stage 0: Dataset Generation          ✅ 완료
Stage 1: Noise Classification        🔄 진행 중 (90%)
Stage 2: LoRA Training               🔜 예정
Stage 3: DNA-MultiLoRA Framework     🔜 예정
```

---

## ✅ Stage 0: 데이터셋 생성 (완료)

### 완료 사항
- ✅ 10가지 노이즈 증강 타입 구현 완료
- ✅ Speaker-disjoint split 구현 (train:dev:eval = 5:2:3)
- ✅ 데이터셋 생성 파이프라인 완료
- ✅ Protocol 파일 생성 완료

### 노이즈 클래스 (10개)

1. `clean` - 원본
2. `background_noise` - 배경 잡음
3. `background_music` - 배경 음악
4. `gaussian_noise` - 가우시안 노이즈
5. `band_pass_filter` - 대역 통과 필터
6. `echo` - 에코
7. `pitch_shift` - 피치 변조
8. `time_stretch` - 속도 변조
9. `reverberation` - 잔향
10. `auto_tune` - 오토튠

### 데이터 통계
- **Total samples**: ~50,000+
- **Split ratio**: 5:2:3 (train:dev:eval)
- **Audio format**: 16kHz mono WAV
- **Duration**: 10초 고정 (padding/clipping)

---

## 🔄 Stage 1: 노이즈 분류 모델 (진행 중)

### 완료된 작업

#### 1. 모델 구현 (11개 모델)
✅ **Transformer 계열**
- AST (Audio Spectrogram Transformer)
- SSAST (Self-Supervised AST) - 3 variants (tiny, small, base)
- HTSAT (Hierarchical Token-Semantic Audio Transformer)
- DASS (DeiT-based Audio Spectrogram Transformer)

✅ **Self-Supervised Learning 계열**
- HuBERT
- Wav2Vec2
- CLAP (Contrastive Language-Audio Pretraining)

✅ **CNN+RNN 계열**
- CNN8RNN (PANN-based)
- CNNLSTM (baseline)
- CNNLSTM_2FF (Spec + F0 fusion)
- CNNLSTM_3FF (Spec + MFCC + F0 fusion)
- **CNNLSTM_3FF_Interaction** (새로 추가 - 2025-03-25)
- **CNNLSTM_3FF_Weight** (새로 추가 - 2025-03-25)

#### 2. 학습 완료된 모델 (10개)
| 모델 | 체크포인트 크기 | 상태 |
|------|----------------|------|
| cnn8rnn | 24 MB | ✅ 완료 |
| cnnlstm | 5.2 MB | ✅ 완료 |
| cnnlstm_2ff | 8.7 MB | ✅ 완료 |
| cnnlstm_3ff | 12 MB | ✅ 완료 |
| htsat | 112 MB | ✅ 완료 |
| hubert | 361 MB | ✅ 완료 |
| ssast_tiny | 23 MB | ✅ 완료 |
| ssast_small | 86 MB | ✅ 완료 |
| ssast_base | 333 MB | ✅ 완료 |
| wav2vec2 | 361 MB | ✅ 완료 |

#### 3. 학습 진행 중/대기 중
- ⏳ **cnnlstm_3ff_interaction**: 학습 완료 (best.pth 29MB)
- ⏳ **cnnlstm_3ff_weight**: 재학습 대기 중 (I/O 에러 해결 완료)

### 진행 중인 작업

#### 최근 작업 (2025-03-25)
1. ✅ CNN-LSTM 3FF 변형 모델 2개 추가
   - Interaction Layer 방식
   - Feature-Level Attention 방식

2. ✅ train.py 통합 완료
   - 모델 factory 함수 추가
   - 데이터 로더 통합
   - Forward 함수 통합

3. ✅ Config 파일 생성
   - `config/cnnlstm_3ff_interaction.yaml`
   - `config/cnnlstm_3ff_weight.yaml`

4. ✅ 모델 저장 I/O 에러 해결
   - 손상된 체크포인트 제거
   - 재학습 준비 완료

### 다음 단계 (TODO)
- [ ] `cnnlstm_3ff_weight` 재학습 실행
- [ ] 전체 모델 성능 비교표 작성
- [ ] 최종 모델 선정 (Stage 2로 전달)
- [ ] AudioMamba 통합 검토 (현재 리포지토리 불완전)

---

## 🔄 Stage 2: LoRA Fine-tuning (진행 중)

### 개요
Audio Deepfake Detection 사전학습 모델에 LoRA 어댑터를 적용하여 노이즈 환경에 적응

### 완료된 작업 (2026-03-26)

#### 1. 프로젝트 구조 재구성 ✅
- ❌ Lightning-hydra 의존성 제거
- ❌ SSL_Anti-spoofing, tcm_add 원본 제거 예정
- ✅ 필요한 모델 코드만 추출하여 독립 구조로 재구성
- ✅ `.agents/rules/noiseclassification.md` 규칙 준수

#### 2. 모델 구현 ✅
**AASIST (`model/aasist.py`):**
- XLSR frontend + Graph Attention Networks
- Spectro-Temporal feature fusion
- LoRA target modules: q_proj, v_proj, k_proj, out_proj, fc1, fc2, LL

**ConformerTCM (`model/conformertcm.py`):**
- XLSR frontend + Conformer encoder
- Head Token Attention mechanism
- LoRA target modules: 동일

**LoRA Wrapper (`model/lora_wrapper.py`):**
- PEFT 라이브러리 기반
- LoRA config: r=8, alpha=16, dropout=0.0

#### 3. 데이터 로더 ✅
- `datautils/dataset.py`: ASVspoof 데이터셋
- 프로토콜 기반 로딩
- 16kHz mono, 64000 samples (4초)

#### 4. Config 파일 생성 ✅
**AASIST:**
- `config/aasist/single_lora.yaml`
- `config/aasist/multi_lora/g0.yaml ~ g7.yaml`

**ConformerTCM:**
- `config/conformertcm/single_lora.yaml`
- `config/conformertcm/multi_lora/g0.yaml ~ g7.yaml`

#### 5. 학습 및 평가 스크립트 ✅
- `train.py`: YAML config 기반 학습
- `eval.py`: EER, Accuracy 평가
- `scripts/aasist/train_single.sh`, `train_multi.sh`
- `scripts/conformertcm/train_single.sh`, `train_multi.sh`
- `scripts/run_all.sh`: 전체 자동 실행

#### 6. 문서화 ✅
- `README.md`: 사용 가이드
- `DEVELOPMENT.md`: 개발 로그
- `requirements.txt`: 의존성 목록

### 진행 중인 작업

#### 1. 사용자 설정 필요 ⏳
- [ ] XLSR 모델 경로 업데이트 (config/*.yaml)
- [ ] 프로토콜 파일 준비 (g0-g7, all_noise)
- [ ] 데이터 경로 업데이트

#### 2. Base Model 준비
- **AASIST**
  - 위치: `pretrained/aasist.pth`
  - 크기: ~1.2GB

- **ConformerTCM**
  - 위치: `pretrained/conformertcm.pth`
  - 크기: ~1.2GB

#### 2. 8-Domain 그룹 정의

Stage 1의 10-class 노이즈를 8개 도메인으로 그룹화:

| Group ID | Domain | 포함 노이즈 클래스 |
|----------|--------|------------------|
| **g0** | Clean | clean |
| **g1** | Background | background_music + background_noise |
| **g2** | AutoTune | auto_tune |
| **g3** | BandPass | band_pass_filter |
| **g4** | Echo | echo |
| **g5** | PitchTime | pitch_shift + time_stretch |
| **g6** | Gaussian | gaussian_noise |
| **g7** | Reverb | reverberation |

#### 3. LoRA 학습 전략
- **Single LoRA**: 전체 노이즈 데이터로 하나의 어댑터 학습
- **Multi-LoRA**: 8개 도메인별 독립 어댑터 학습
- 상세 내용은 `02_LoRA_Training/` 참고

### 예상 일정
- **시작 시기**: Stage 1 완료 직후 (2025-03-26~)
- **소요 시간**: 2-3주 예상

---

## 🔜 Stage 3: DNA-MultiLoRA Framework (예정)

### 계획된 통합 시스템

```
입력 음성
    ↓
[Noise Classifier]  ← Stage 1의 최고 성능 모델
    ↓
10-class 예측 → 8-domain 매핑
    ↓
[LoRA Router]  ← 도메인 그룹 기반 어댑터 선택
    ↓
    ├─ g0: Clean LoRA
    ├─ g1: Background LoRA
    ├─ g2: AutoTune LoRA
    ├─ g3: BandPass LoRA
    ├─ g4: Echo LoRA
    ├─ g5: PitchTime LoRA
    ├─ g6: Gaussian LoRA
    └─ g7: Reverb LoRA
    ↓
[Base Model + Selected LoRA]  ← XLSR + Conformer/AASIST
    ↓
Deepfake 탐지 결과 (Bonafide/Spoof)
```

### 핵심 기능
1. **10-to-8 Domain Mapping**
   - Noise Classifier 10-class 출력
   - 8-domain 그룹으로 매핑
   - 예: `background_music` OR `background_noise` → `g1 (Background)`

2. **Dynamic Adapter Selection**
   - 도메인 예측 결과 기반 LoRA 선택
   - 실시간 어댑터 스위칭
   - Config: `xlsr_conformertcm_mul_lora.yaml` 참고

3. **Multi-view Ensemble** (선택적)
   - 여러 도메인 혼합 시 가중 평균
   - Soft routing 메커니즘

4. **End-to-End 평가**
   - ASVspoof5 데이터셋 평가
   - EER, t-DCF 메트릭

---

## 📊 주요 의사결정 기록

### 2025-03-25
- **결정**: 8-Domain 그룹 정의
  - **목적**: Stage 2 Multi-LoRA 학습 전략 수립
  - **매핑 원칙**:
    - 유사한 특성의 노이즈 통합 (g1: background, g5: pitch/time)
    - 독립적 특성은 별도 그룹 (g2: autotune, g4: echo, etc.)
  - **기대효과**:
    - 도메인별 특화된 어댑터 학습
    - 추론 시 효율적인 어댑터 선택

- **결정**: AudioMamba 통합 보류
  - **이유**: 리포지토리 불완전 (모델 코드 누락)
  - **대안**: 현재 11개 모델로 충분히 다양한 접근 시도
  - **향후**: 저자가 코드 공개 시 재검토 가능

- **결정**: CNN-LSTM 3FF 변형 2개 추가
  - **이유**: Feature fusion 방법론 다양화
  - **방법 1**: Interaction Layer (1536→1024→1536)
  - **방법 2**: Feature-Level Attention (adaptive weights)
  - **기대효과**: Multi-modal fusion 성능 향상

- **결정**: GELU 활성화 함수 사용
  - **이유**: Transformer 계열에서 우수한 성능
  - **적용 모델**: 3ff_interaction, 3ff_weight

### 2025-03-20~24
- **결정**: 10개 모델 우선 학습 완료
  - **목적**: 다양한 아키텍처 성능 비교
  - **범위**: Transformer, SSL, CNN+RNN 계열 모두 포함

---

## 🔗 관련 문서

- **상세 개발 로그**: `01_Noise_Classification/DEVELOPMENT.md`
- **프로젝트 README**: `README.md`
- **Agent 규칙**: `.agents/rules/noiseclassification.md`
- **Stage 2 (LoRA)**: `02_LoRA_Training/` 디렉토리 참고

---

## 📝 노트

### Stage 1 → Stage 2 전환 조건
1. ✅ 전체 모델 학습 완료
2. ⏳ 성능 비교표 작성
3. ⏳ 최고 성능 Noise Classifier 선정
4. ⏳ Stage 2 데이터셋 준비 확인 (8-domain 재분류)

### 예상 타임라인
- **Stage 1 완료**: 2025-03-26 (예상)
- **Stage 2 시작**: 2025-03-27 (예상)
  - Single LoRA 학습: 3-5일
  - Multi-LoRA 학습 (x8): 1-2주
- **Stage 2 완료**: 2025-04-15 (예상)
- **Stage 3 통합**: 2025-04-16~ (예상)
- **논문 작성**: 2025-05~ (예상)

---

**Last Updated**: 2025-03-25 17:20 KST
