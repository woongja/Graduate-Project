# 02_LoRA_Training

Audio Deepfake Detection 모델에 도메인별 LoRA 학습 + 평가 + 자동 HP 탐색

---

## ADD 모델 (4종)

| 모델 | 파일 | Pretrained | 비고 |
|---|---|---|---|
| AASIST | `model/aasist.py` | `pretrained/aasist.pth` | XLSR + Graph ATT |
| ConformerTCM | `model/conformertcm.py` | `pretrained/conformertcm.pth` | XLSR + Conformer, config에 emb_size 필요 |
| XLSR-SLS | `model/xlsr_sls.py` | `pretrained/XLSR-SLS.pth` | XLSR + Spectral Level Stats |
| XLSR-Mamba | `model/xlsr_mamba.py` | `pretrained/XLSR-Mamba-DF/model.safetensors` | XLSR + Bidirectional Mamba, mamba_blocks.py 의존 |

---

## 도메인 매핑 (10개)

| Domain | aug_type |
|--------|----------|
| D0 | clean |
| D1 | background_noise |
| D2 | background_music |
| D3 | auto_tune |
| D4 | bandpass (high+low) |
| D5 | echo |
| D6 | pitch_shift |
| D7 | time_stretch |
| D8 | gaussian_noise |
| D9 | reverberation |

---

## 핵심 파일

| 파일 | 역할 |
|---|---|
| `main.py` | 학습 + 평가 통합 (`--eval` 평가, 없으면 학습) |
| `model/lora_wrapper.py` | `apply_lora()` LoRA 유틸 |
| `datautils/data_utils.py` | 데이터 로더, `genDomain_list()` 도메인 필터링 |
| `evaluate_metrics.py` | `compute_eer()` EER 계산 |
| `show_results.py` | SQLite 대시보드 |

---

## 실행 방법

### 1. Pretrained 모델 평가

```bash
# 원본 데이터셋
bash scripts/eval.sh aasist base asv19
bash scripts/eval.sh conformertcm base df21
bash scripts/eval.sh xlsr_sls base itw
bash scripts/eval.sh xlsr_mamba base asv19
```

### 2. 도메인별 LoRA 학습 (단일)

```bash
# D1 LoRA 학습
bash scripts/train_lora.sh aasist 1
```

### 3. 자동 HP 탐색 (Optuna)

```bash
# 1-step 파이프라인 테스트
python scripts/auto_lora/test.py --gpu MIG-xxx --model aasist

# 특정 모델 HP 탐색
python scripts/auto_lora/runner.py --model aasist --gpu MIG-xxx --domain 1 --n_trials 50

# 4모델 병렬 HP 탐색 (start.sh에서 GPU 설정)
bash scripts/auto_lora/start.sh
```

### 4. 도메인별 LoRA 전체 학습

```bash
# config.py에서 모델/HP/GPU 설정 후
python scripts/domain_lora/generate.py --clean
bash scripts/domain_lora/start.sh
bash scripts/domain_lora/watch.sh  # 대시보드
```

### 5. 결과 대시보드

```bash
python show_results.py
```

---

## 디렉토리 구조

```
02_LoRA_Training/
├── main.py                         # 학습 + 평가 통합
├── model/
│   ├── aasist.py                   # AASIST
│   ├── conformertcm.py             # ConformerTCM
│   ├── xlsr_sls.py                 # XLSR-SLS
│   ├── xlsr_mamba.py               # XLSR-Mamba
│   ├── mamba_blocks.py             # Mamba 블록 (xlsr_mamba 의존)
│   ├── lora_wrapper.py             # LoRA 유틸
│   └── conformer.py                # Conformer 블록
├── config/
│   ├── aasist_baseline.yaml
│   ├── conformertcm_baseline.yaml
│   ├── xlsr_sls_baseline.yaml
│   ├── xlsr_mamba_baseline.yaml
│   └── *_multi_lora_g*.yaml        # 도메인별 LoRA config
├── datautils/
│   ├── data_utils.py               # 데이터 로더 + 도메인 필터링
│   └── RawBoost.py
├── scripts/
│   ├── eval.sh                     # 단일 평가
│   ├── train.sh                    # 단일 학습
│   ├── train_lora.sh               # 단일 LoRA 학습
│   ├── auto_lora/                  # Optuna 자동 HP 탐색
│   │   ├── runner.py               # Optuna 기반 탐색
│   │   ├── test.py                 # 1-step 테스트
│   │   ├── start.sh                # 4모델 병렬 시작
│   │   └── program.md              # 탐색 지침
│   ├── domain_lora/                # 도메인별 LoRA 학습
│   │   ├── config.py               # 모델/HP/GPU 설정
│   │   ├── generate.py             # 실험 생성
│   │   ├── scheduler.py            # GPU 스케줄러
│   │   └── dashboard.py            # 대시보드
│   └── hp_search/                  # HP grid search (legacy)
├── pretrained/                     # Pretrained 체크포인트
├── protocols/                      # 프로토콜 파일
├── results/                        # 평가 결과 + dashboard.db
├── out/                            # 학습 출력
│   ├── auto_lora/                  # Optuna 탐색 결과
│   ├── domain_lora/                # 도메인 LoRA 결과
│   └── hp_search/                  # HP search 결과
├── evaluate_metrics.py             # EER 계산
└── show_results.py                 # 결과 대시보드
```

---

## LoRA 최적 HP

| 모델 | lr | r | alpha | 비고 |
|---|---|---|---|---|
| AASIST | 5e-5 | 32 | 64 | HP search 완료 |
| ConformerTCM | 1e-4 | 16 | 32 | HP search 완료 |
| XLSR-SLS | - | - | - | auto_lora로 탐색 필요 |
| XLSR-Mamba | - | - | - | auto_lora로 탐색 필요 |

target_modules (공통): `[q_proj, v_proj, k_proj, out_proj, fc1, fc2, LL]`

---

## 프로토콜 형식

| 파일 | 형식 | 용도 |
|---|---|---|
| `protocols/original/*` | `key subset label` | 원본 데이터 eval |
| `protocols/asv19_*.txt` | `abs_path label aug_type` | NC 데이터 학습/eval |
