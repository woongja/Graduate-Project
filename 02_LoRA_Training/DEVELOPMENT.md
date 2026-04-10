# Stage 2 LoRA Training — Development Log

> 개발 일지 및 코드 변경 사항 기록

---

## 📅 2026-03-26: 프로젝트 구조 재구성 시작

### 목표
- Lightning-hydra 의존성 제거
- AASIST, ConformerTCM 모델을 독립 실행 가능한 구조로 재구성
- `.agents/rules/noiseclassification.md` 규칙 준수

### 작업 내용

#### 1. 모델 코드 추출 및 재구성

**1.1 XLSR Frontend 추출 (`model/xlsr_frontend.py`)**
- SSL_Anti-spoofing와 tcm_add의 `SSLModel` 통합
- Fairseq 기반 XLSR feature extractor
- 출력 차원: 1024

**1.2 AASIST 모델 (`model/aasist.py`)**
- 출처: `SSL_Anti-spoofing/model.py`
- 추출 컴포넌트:
  - `GraphAttentionLayer`: 그래프 어텐션
  - `HtrgGraphAttentionLayer`: 이종 그래프 어텐션
  - `GraphPool`: 그래프 풀링
  - `Residual_block`: ResNet 기반 인코더
  - `AASIST`: 메인 모델
- 제거 항목:
  - 평가 스크립트 (`eval_metric_LA.py`, `eval_metrics_DF.py`)
  - 데이터베이스 폴더 (`database/`)
  - fairseq 서브모듈

**1.3 ConformerTCM 모델 (`model/conformertcm.py`)**
- 출처: `tcm_add/model.py` + `tcm_add/conformer.py`
- 통합 컴포넌트:
  - `ConformerBlock`: Conformer 인코더 블록
  - `Attention`: Head Token Attention
  - `FeedForward`, `ConformerConvModule`: Conformer 구성 요소
  - `MyConformer`: Conformer 인코더
  - `ConformerTCM`: 메인 모델
- 하이퍼파라미터:
  - `emb_size`: 144
  - `num_encoders`: 4
  - `heads`: 4
  - `kernel_size`: 31

**1.4 LoRA Wrapper (`model/lora_wrapper.py`)**
- PEFT 라이브러리 기반 LoRA 적용
- 기능:
  - `apply_lora()`: LoRA config 기반 어댑터 적용
  - `load_lora_weights()`: 체크포인트 로드
  - `merge_and_unload_lora()`: LoRA 가중치 병합
  - `freeze_base_model()`: Base model freeze
  - `print_model_info()`: 파라미터 정보 출력

#### 2. 데이터 로더 구현

**`datautils/dataset.py`**
- `ASVspoofDataset`: 프로토콜 기반 데이터셋
  - 프로토콜 형식: `file_path bonafide/spoof` 또는 `file_path 0/1`
  - 16kHz mono, 64000 samples (4초)
  - Repeat padding/Random crop
- `collate_fn`: DataLoader용 collate 함수

#### 3. Config 파일 생성

**AASIST Configs:**
- `config/aasist/single_lora.yaml`: Single LoRA (전체 노이즈)
- `config/aasist/multi_lora/g0.yaml ~ g7.yaml`: 8-domain LoRA

**ConformerTCM Configs:**
- `config/conformertcm/single_lora.yaml`: Single LoRA
- `config/conformertcm/multi_lora/g0.yaml ~ g7.yaml`: 8-domain LoRA

**주요 설정:**
```yaml
model:
  lora:
    r: 8
    alpha: 16
    target_modules: [q_proj, v_proj, k_proj, out_proj, fc1, fc2, LL]

training:
  max_epochs: 30
  lr: 0.000003
  weight_decay: 0.0001
  cross_entropy_weight: [0.9, 0.1]
```

#### 4. 학습 및 평가 스크립트

**`train.py`:**
- YAML config 기반 학습
- 모델 factory 함수 (`create_model`)
- LoRA 적용 후 학습
- TensorBoard 로깅
- Early stopping (patience=5)
- Best/Last checkpoint 저장

**`eval.py`:**
- 학습된 LoRA 모델 평가
- 메트릭: Accuracy, EER (Equal Error Rate)
- Confusion Matrix, Classification Report
- 결과 파일 저장

**학습 스크립트:**
- `scripts/aasist/train_single.sh`: AASIST Single LoRA
- `scripts/aasist/train_multi.sh`: AASIST Multi-LoRA (g0-g7)
- `scripts/conformertcm/train_single.sh`: ConformerTCM Single LoRA
- `scripts/conformertcm/train_multi.sh`: ConformerTCM Multi-LoRA
- `scripts/run_all.sh`: 전체 실험 자동 실행

---

## 📦 디렉토리 구조 변경

### 이전 (Lightning-hydra 기반)
```
02_LoRA_Training/
├── Lightning-hydra/
│   ├── configs/
│   ├── src/
│   │   ├── models/
│   │   ├── data/
│   │   └── ...
│   ├── scripts/icassp26/
│   └── ...
├── SSL_Anti-spoofing/
└── tcm_add/
```

### 이후 (깔끔한 독립 구조)
```
02_LoRA_Training/
├── model/
├── datautils/
├── config/
├── scripts/
├── train.py
├── eval.py
├── pretrained/
└── out/
```

---

## 🔧 주요 의사결정

### 1. 왜 Lightning-hydra를 제거했는가?
- **문제점:**
  - 과도하게 복잡한 구조 (10+ 디렉토리)
  - 불필요한 의존성 (hydra, lightning, wandb 등)
  - 실험 재현 어려움
- **해결책:**
  - 필요한 컴포넌트만 추출
  - 단순한 YAML config + Python 스크립트
  - 의존성 최소화

### 2. 모델 구조 설계
- **XLSR Frontend 공통화:**
  - AASIST와 ConformerTCM 모두 XLSR 사용
  - `XLSRFrontend` 클래스로 통합
- **LoRA 적용 방식:**
  - PEFT 라이브러리 사용 (표준 방식)
  - Target modules: attention QKV, FFN, projection layers

### 3. Config 파일 설계
- **단순화:**
  - 파일명: `g0.yaml` ~ `g7.yaml` (너무 긴 이름 제거)
  - 경로: `config/aasist/multi_lora/` (stage2 제거)
- **중복 제거:**
  - Base config 상속 대신 각 파일 독립 작성

---

## 🚧 해결한 문제들

### 문제 1: AASIST 모델 import 에러
- **원인:** SSL_Anti-spoofing의 상대 import
- **해결:** 모든 코드를 하나의 파일로 통합

### 문제 2: ConformerTCM의 einops 의존성
- **원인:** `conformer.py`에서 einops 사용
- **해결:** `requirements.txt`에 einops 추가

### 문제 3: LoRA target modules 불일치
- **원인:** AASIST와 ConformerTCM의 layer 이름 차이
- **해결:** 공통 모듈 명시 (`q_proj`, `v_proj`, `k_proj` 등)

---

## 📊 코드 통계

### 파일 수
- 모델 파일: 4개 (`aasist.py`, `conformertcm.py`, `xlsr_frontend.py`, `lora_wrapper.py`)
- Config 파일: 20개 (2 models × 10 configs)
- 스크립트: 5개 (학습 스크립트)

### 코드 라인 수
- `model/aasist.py`: ~470 lines
- `model/conformertcm.py`: ~330 lines
- `model/xlsr_frontend.py`: ~60 lines
- `model/lora_wrapper.py`: ~120 lines
- `train.py`: ~200 lines
- `eval.py`: ~160 lines

### 제거된 코드
- Lightning-hydra: ~10,000+ lines
- SSL_Anti-spoofing: ~2,000+ lines
- tcm_add: ~500+ lines
- **총 제거:** ~12,500+ lines

---

## 🎯 다음 작업

### 완료 후 작업
1. ✅ 모델 코드 추출 완료
2. ✅ 데이터 로더 구현 완료
3. ✅ Config 파일 생성 완료
4. ✅ train.py, eval.py 구현 완료
5. ✅ 학습 스크립트 생성 완료
6. ✅ DEVELOPMENT.md 작성 완료

### 남은 작업
1. ⏳ PROGRESS.md 업데이트
2. ⏳ 원본 폴더 삭제 (Lightning-hydra, SSL_Anti-spoofing, tcm_add)
3. ⏳ 사용자: XLSR 경로 업데이트 (`config/*/single_lora.yaml`, `config/*/multi_lora/*.yaml`)
4. ⏳ 사용자: 프로토콜 파일 준비 (8-domain + all_noise)
5. ⏳ 실제 학습 실행

---

## 📝 참고사항

### XLSR 모델 경로
- 사용자가 업데이트 필요: `/path/to/xlsr2_300m.pt`
- 모든 config 파일에 공통 적용

### 프로토콜 파일 형식
```
/path/to/audio1.wav bonafide
/path/to/audio2.wav spoof
/path/to/audio3.wav 0
/path/to/audio4.wav 1
```

### 사전학습 모델
- `pretrained/aasist.pth`: AASIST 베이스 모델
- `pretrained/conformertcm.pth`: ConformerTCM 베이스 모델

---

**Last Updated**: 2026-03-26 11:42 KST
