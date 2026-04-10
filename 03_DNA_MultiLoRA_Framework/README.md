# 03_DNA_MultiLoRA_Framework

노이즈 분류 모듈(NC)과 도메인별 LoRA를 결합한 오디오 딥페이크 탐지 프레임워크.

## 파이프라인

```
Audio → [노이즈 분류기] → Domain ID (D0~D7)
                              ↓
       [Pretrained ADD] + [Domain LoRA] → Spoof/Bonafide Score
```

- D0 (clean): LoRA 없이 base model 사용
- D1~D7: 해당 도메인에 최적화된 LoRA 적용

## 디렉토리 구조

```
03_DNA_MultiLoRA_Framework/
├── config/
│   └── pipeline.yaml          # NC/ADD/LoRA/데이터셋 경로 설정
├── model/
│   ├── noise_classifier.py    # 노이즈 분류 모델 래퍼
│   └── add_model.py           # ADD 모델 + LoRA 스위칭 래퍼
├── scripts/
│   └── eval_all.sh            # 전체 조합 자동 평가 (10조합 × 2데이터셋)
├── eval.py                    # 메인 평가 스크립트
├── dashboard.py               # 결과 대시보드
├── results/                   # 평가 결과 저장
└── README.md
```

## 사전 준비

- `01_Noise_Classification/out/` — 학습된 노이즈 분류 모델 (best.pth)
- `02_LoRA_Training/pretrained/` — AASIST, ConformerTCM pretrained
- `02_LoRA_Training/out/hp_search/` — D1 LoRA 체크포인트
- `02_LoRA_Training/out/domain_lora/` — D2~D7 LoRA 체크포인트

## 실행 방법

### 1. 단일 조합 평가

```bash
# NC=CNN8RNN-3FF-CrossModal, ADD=AASIST, Dataset=ASV19
python eval.py --nc cnn8rnn_3ff_crossmodal --add aasist --dataset asv19

# NC=SSAST-tiny, ADD=ConformerTCM, Dataset=DF21
python eval.py --nc ssast_tiny --add conformertcm --dataset df21

# GPU 지정
python eval.py --nc htsat --add aasist --dataset asv19 --gpu MIG-xxxx
```

### 2. 전체 조합 자동 평가

```bash
# NC 5종 × ADD 2종 × Dataset 2종 = 20개 평가
bash scripts/eval_all.sh

# GPU 지정
bash scripts/eval_all.sh MIG-xxxx
```

### 3. 결과 대시보드

```bash
# 터미널에 결과 테이블 출력
python dashboard.py

# CSV로 내보내기
python dashboard.py --csv results/summary.csv
```

## 노이즈 분류 모듈 (5종)

| 이름 | 모듈 | 체크포인트 |
|------|------|-----------|
| CNN8RNN-3FF-CrossModal | `cnn8rnn_3ff_crossmodal` | `01_NC/out/cnn8rnn_3ff_crossmodal/best.pth` |
| CNN8RNN-3FF | `cnn8rnn_3ff_base` | `01_NC/out/cnn8rnn_3ff_base/best.pth` |
| CNN+LSTM | `cnnlstm` | `01_NC/out/cnnlstm/best.pth` |
| SSAST-tiny | `ssast_tiny` | `01_NC/out/ssast_tiny_patch_400/best.pth` |
| HTS-AT | `htsat` | `01_NC/out/htsat/best.pth` |

## ADD 모델 + LoRA (2종)

| 모델 | LoRA HP | 비고 |
|------|---------|------|
| AASIST | lr=5e-5, r=32, alpha=64 | D1~D7 독립 LoRA |
| ConformerTCM | lr=1e-4, r=16, alpha=32 | D1~D7 독립 LoRA |

## 도메인 매핑 (10클래스 → 7도메인)

| NC 출력 | 도메인 |
|---------|--------|
| 0: clean | D0 (LoRA 없음) |
| 1: bg_noise, 2: bg_music | D1 |
| 9: auto_tune | D2 |
| 4: bandpass | D3 |
| 5: echo | D4 |
| 6: pitch_shift, 7: time_stretch | D5 |
| 3: gaussian | D6 |
| 8: reverberation | D7 |

## 설정 변경

`config/pipeline.yaml`에서:
- 노이즈 분류 모듈 경로 변경
- ADD 모델/LoRA 체크포인트 경로 변경
- 평가 데이터셋 프로토콜 경로 변경
