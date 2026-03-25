# Wav2Vec2-base-AudioSet Fine-tuning Wrapper
# Pretrained: ALM/wav2vec2-base-audioset (AudioSet SSL)
# transformers.Wav2Vec2Model backbone + mean-pool + classification head

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2Classifier(nn.Module):
    """
    Wav2Vec2 기반 음성 분류 모델.

    사용법:
        model = Wav2Vec2Classifier(
            num_classes=10,
            pretrained_path='wav2vec2-base-audioset'   # 로컬 경로 or HF repo id
        )
        logits = model(waveform)   # waveform: (B, num_samples) @ 16 kHz

    Pooling:
        - 'mean'      : 전체 프레임 평균 (기본값)
        - 'first'     : 첫 번째 토큰
        - 'attention' : 학습 가능한 attention weight로 weighted sum
    """

    SAMPLE_RATE = 16000

    def __init__(self,
                 num_classes=10,
                 pretrained_path='wav2vec2-base-audioset',
                 pooling='mean',
                 dropout=0.1,
                 freeze_feature_extractor=True):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_path)
        hidden_size   = self.wav2vec2.config.hidden_size  # 768

        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        self.pooling = pooling
        if pooling == 'attention':
            self.attn_weight = nn.Linear(hidden_size, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def _pool(self, hidden_states):
        """(B, T, D) → (B, D)"""
        if self.pooling == 'mean':
            return hidden_states.mean(dim=1)
        elif self.pooling == 'first':
            return hidden_states[:, 0, :]
        elif self.pooling == 'attention':
            w = torch.softmax(self.attn_weight(hidden_states), dim=1)
            return (hidden_states * w).sum(dim=1)

    def forward(self, waveform):
        """
        Args:
            waveform : (B, num_samples) — raw audio @ 16 kHz, float32
        Returns:
            logits   : (B, num_classes)
        """
        outputs = self.wav2vec2(input_values=waveform)
        hidden  = outputs.last_hidden_state   # (B, T, 768)
        pooled  = self._pool(hidden)          # (B, 768)
        return self.classifier(pooled)        # (B, num_classes)

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE
