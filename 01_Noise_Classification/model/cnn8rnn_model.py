# CNN8RNN for Noise Classification
# Pretrained: marsyas/cnn8rnn-audioset-sed (AudioSet SED)
# CNN8 + BiGRU backbone + classification head

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms


# ── Utility ───────────────────────────────────────────────────────────────────

def _init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

def _init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3), (1,1), (1,1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3,3), (1,1), (1,1), bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1); _init_layer(self.conv2)
        _init_bn(self.bn1);      _init_bn(self.bn2)

    def forward(self, x, pool_size=(2,2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg+max':
            x = F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, pool_size)
        else:
            x = F.max_pool2d(x, pool_size)
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class CNN8RNNClassifier(nn.Module):
    """
    CNN8RNN 기반 noise classification.
    AudioSet SED pretrained backbone → fc_audioset 교체 → fine-tune

    사용법:
        model = CNN8RNNClassifier(
            num_classes=10,
            pretrained_path='cnn8rnn-audioset-sed'
        )
        logits = model(waveform)   # waveform: (B, num_samples) @ 32 kHz
    """

    SAMPLE_RATE = 32000

    def __init__(self,
                 num_classes=10,
                 pretrained_path=None,
                 dropout=0.5):
        super().__init__()

        # ── Audio front-end (torchaudio, 동일 설정 유지) ─────────────────────
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=32000, n_fft=1024, win_length=1024,
            hop_length=320, f_min=50, f_max=14000,
            n_mels=64, norm='slaney', mel_scale='slaney')
        self.db_transform = transforms.AmplitudeToDB()

        # ── Backbone ─────────────────────────────────────────────────────────
        self.bn0          = nn.BatchNorm2d(64)
        self.conv_block1  = ConvBlock(1,   64)
        self.conv_block2  = ConvBlock(64,  128)
        self.conv_block3  = ConvBlock(128, 256)
        self.conv_block4  = ConvBlock(256, 512)
        self.fc1          = nn.Linear(512, 512, bias=True)
        self.rnn          = nn.GRU(512, 256, bidirectional=True, batch_first=True)

        # ── Classification head (pretrained: 447 → 우리: num_classes) ────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        _init_bn(self.bn0)
        _init_layer(self.fc1)

        # ── Load pretrained weights ───────────────────────────────────────────
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        """backbone 가중치만 로드, fc_audioset(head)는 skip"""
        print(f'Loading CNN8RNN pretrained: {path}')
        import os

        # safetensors 우선, 없으면 pytorch_model.bin
        st_path  = os.path.join(path, 'model.safetensors')
        bin_path = os.path.join(path, 'pytorch_model.bin')

        if os.path.exists(st_path):
            from safetensors.torch import load_file
            sd = load_file(st_path)
        elif os.path.exists(bin_path):
            sd = torch.load(bin_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'No model weights found in {path}')

        # 헤드(fc_audioset) 제외하고 로드
        sd_backbone = {
            k: v for k, v in sd.items()
            if not k.startswith('fc_audioset') and not k.startswith('temporal_pooling')
        }
        missing, unexpected = self.load_state_dict(sd_backbone, strict=False)
        print(f'  Loaded  — missing: {len(missing)}, unexpected: {len(unexpected)}')

    def _extract_features(self, waveform):
        """waveform (B, T) → frame-level features (B, T', 512)"""
        x = self.melspec_extractor(waveform)   # (B, 64, T)
        x = self.db_transform(x)
        x = x.transpose(1, 2).unsqueeze(1)     # (B, 1, T, 64)

        x = x.transpose(1, 3)                  # (B, 64, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)                  # (B, 1, T, 64)

        x = self.conv_block1(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)               # (B, 512, T')
        x = x.transpose(1, 2)                  # (B, T', 512)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))               # (B, T', 512)
        x, _ = self.rnn(x)                     # (B, T', 512)
        return x

    def forward(self, waveform):
        """
        Args:
            waveform : (B, num_samples) — raw audio @ 32 kHz
        Returns:
            logits   : (B, num_classes)
        """
        x = self._extract_features(waveform)   # (B, T', 512)
        x = x.mean(dim=1)                      # (B, 512) — temporal mean pool
        return self.head(x)                    # (B, num_classes)

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE
