# 02_LoRA_Training - Training Guide

## 📌 학습 전략

### 1️⃣ **Fine-tuning** (전체 모델 학습)
- LoRA 없이 사전학습 모델을 전체 fine-tuning
- 모든 파라미터 업데이트
- **용도**: Baseline 성능 확인

### 2️⃣ **Single LoRA** (전역 LoRA)
- 전체 노이즈 데이터로 하나의 LoRA 어댑터 학습
- 파라미터 효율적
- **용도**: LoRA baseline

### 3️⃣ **Multi-LoRA** (도메인별 LoRA)
- 8개 노이즈 도메인별 독립 LoRA 학습
- 각 도메인에 특화된 어댑터
- **용도**: 최종 DNA-MultiLoRA 시스템

---

## 🚀 학습 순서

```
1. Fine-tuning (Baseline)
   ↓
2. Single LoRA (LoRA Baseline)
   ↓
3. Multi-LoRA (Domain-specific)
   ↓
4. Stage 3: DNA-MultiLoRA Framework
```

---

## 📊 비교 실험

| 방법 | 학습 파라미터 | 메모리 | 성능 (예상) |
|------|--------------|--------|------------|
| **Fine-tuning** | 전체 (~100M) | 큰 | High |
| **Single LoRA** | LoRA만 (~1M) | 작음 | Medium-High |
| **Multi-LoRA** | LoRA×8 (~8M) | 작음 | Highest |

---

## 🔧 실행 방법

### A. Fine-tuning
```bash
# AASIST Fine-tuning
bash scripts/train_finetune.sh aasist

# ConformerTCM Fine-tuning
bash scripts/train_finetune.sh conformertcm
```

### B. Single LoRA
```bash
# AASIST Single LoRA
bash scripts/train_single_lora.sh aasist

# ConformerTCM Single LoRA
bash scripts/train_single_lora.sh conformertcm
```

### C. Multi-LoRA (8 domains)
```bash
# AASIST Multi-LoRA
bash scripts/train_multi_lora.sh aasist

# ConformerTCM Multi-LoRA
bash scripts/train_multi_lora.sh conformertcm
```

### D. 전체 실험
```bash
bash scripts/run_all_experiments.sh
```

---

## 📁 출력 디렉토리

```
out/
├── aasist/
│   ├── finetune/          # Fine-tuning 결과
│   ├── single_lora/       # Single LoRA 결과
│   └── multi_lora/        # Multi-LoRA 결과 (g0-g7)
│       ├── g0/
│       ├── g1/
│       └── ...
└── conformertcm/
    ├── finetune/
    ├── single_lora/
    └── multi_lora/
```

---

## 🎯 8-Domain Groups

| Group | Domain | Noise Types |
|-------|--------|-------------|
| g0 | Clean | clean |
| g1 | Background | background_music + background_noise |
| g2 | AutoTune | auto_tune |
| g3 | BandPass | band_pass_filter (high_pass + low_pass) |
| g4 | Echo | echo |
| g5 | PitchTime | pitch_shift + time_stretch |
| g6 | Gaussian | gaussian_noise |
| g7 | Reverb | reverberation |

---

## 📝 다음 단계

1. ✅ Fine-tuning baseline 확보
2. ✅ Single LoRA 성능 비교
3. ✅ Multi-LoRA 도메인별 학습
4. ⏭️ Stage 3: Noise Classifier + Multi-LoRA 통합
