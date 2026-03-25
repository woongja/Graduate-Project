# HuBERT-base-AudioSet Fine-tuning Wrapper
# Pretrained: ALM/hubert-base-audioset (AudioSet SSL)
# transformers.HubertModel backbone + mean-pool + classification head

import torch
import torch.nn as nn
from transformers import HubertModel


class HubertClassifier(nn.Module):
    """
    HuBERT 기반 음성 분류 모델.

    사용법:
        model = HubertClassifier(
            num_classes=10,
            pretrained_path='hubert-base-audioset'   # 로컬 경로 or HF repo id
        )
        logits = model(waveform)   # waveform: (B, num_samples) @ 16 kHz

    Pooling:
        - 'mean' : 전체 프레임 평균 (기본값)
        - 'first': [CLS] 대신 첫 번째 토큰
        - 'attention': 학습 가능한 attention weight로 weighted sum
    """

    SAMPLE_RATE = 16000

    def __init__(self,
                 num_classes=10,
                 pretrained_path='hubert-base-audioset',
                 pooling='mean',
                 dropout=0.1,
                 freeze_feature_extractor=True):
        """
        Args:
            num_classes           : 분류 클래스 수
            pretrained_path       : HuBERT 모델 경로 (로컬 디렉토리 or HF repo id)
            pooling               : 'mean' | 'first' | 'attention'
            dropout               : 분류 헤드 dropout
            freeze_feature_extractor: CNN feature extractor 고정 여부 (권장 True)
        """
        super().__init__()

        self.hubert = HubertModel.from_pretrained(pretrained_path)
        hidden_size = self.hubert.config.hidden_size  # 768

        # CNN feature extractor는 gradient 필요 없음 (표준 fine-tuning 방식)
        if freeze_feature_extractor:
            self.hubert.feature_extractor._freeze_parameters()

        self.pooling = pooling
        if pooling == 'attention':
            self.attn_weight = nn.Linear(hidden_size, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def _pool(self, hidden_states, attention_mask=None):
        """(B, T, D) → (B, D)"""
        if self.pooling == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return hidden_states.mean(dim=1)

        elif self.pooling == 'first':
            return hidden_states[:, 0, :]

        elif self.pooling == 'attention':
            w = torch.softmax(self.attn_weight(hidden_states), dim=1)  # (B, T, 1)
            return (hidden_states * w).sum(dim=1)                       # (B, D)

    def forward(self, waveform, attention_mask=None):
        """
        Args:
            waveform       : (B, num_samples) — raw audio @ 16 kHz, float32
            attention_mask : (B, num_samples) — optional padding mask
        Returns:
            logits         : (B, num_classes)
        """
        outputs = self.hubert(
            input_values=waveform,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state   # (B, T, 768)
        pooled = self._pool(hidden)          # (B, 768)
        return self.classifier(pooled)       # (B, num_classes)

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE
