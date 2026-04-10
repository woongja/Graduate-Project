"""
Domain LoRA Configuration
- 모델별 도메인, HP, GPU 설정을 여기서 수정하세요.
- MODELS 리스트 순서대로 실행됩니다.
"""

# ── 모델별 설정 ──
MODELS = {
    "aasist": {
        "domains": [2, 3, 4, 5, 6, 7],
        "hps": [
            {"lr": 5e-5, "r": 16, "alpha": 32},
            {"lr": 1e-4, "r": 32, "alpha": 32},
            {"lr": 5e-5, "r": 32, "alpha": 64},
        ],
    },
    "conformertcm": {
        "domains": [2, 3, 4, 5, 6, 7],
        "hps": [
            {"lr": 1e-4, "r": 16, "alpha": 32},
            {"lr": 1e-4, "r": 16, "alpha": 16},
            {"lr": 1e-4, "r": 4, "alpha": 8},
        ],
    },
}

# ── 실행할 모델 순서 (aasist 끝나면 conformertcm 자동 시작) ──
RUN_ORDER = ["aasist", "conformertcm"]

# ── GPU 목록 ──
GPUS = [
    "MIG-6e4275af-2db0-51f1-a601-7ad8a1002745",
    "MIG-57de94a5-be15-5b5a-b67e-e118352d8a59",
    "MIG-8cdeef83-092c-5a8d-a748-452f299e1df0",
    "MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494",
    "MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd"
]

# ── 학습 설정 ──
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 64
MAX_EPOCHS = 100
PATIENCE = 5
