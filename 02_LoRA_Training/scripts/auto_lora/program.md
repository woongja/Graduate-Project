# AutoResearch LoRA — 실험 지침

## 목표
각 ADD 모델(aasist, conformertcm, xlsr_sls, xlsr_mamba)에 대해
LoRA 하이퍼파라미터를 자동 탐색하여 특정 도메인의 EER을 최소화.

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

## 탐색 파라미터
- lr: 1e-6 ~ 1e-3 (log scale)
- r: [4, 8, 16, 32]
- alpha: r × [1, 2, 4]
- dropout: 0.0 ~ 0.2
- batch_size: [16, 24, 32]

## 고정 파라미터
- max_epochs: 100
- patience: 5
- target_modules: [q_proj, v_proj, k_proj, out_proj, fc1, fc2, LL]
- weight_decay: 1e-4
- class_weights: [0.9, 0.1]

## 제약
- 한 모델 내 모든 도메인은 동일 LoRA HP 사용
- GPU: MIG 48GB 1개 per model
- 평가: 해당 도메인 ASV19 EER (minimize)

## 실행
```bash
# 테스트
python scripts/auto_lora/test.py --gpu MIG-xxx

# 전체 실행
bash scripts/auto_lora/start.sh
```
