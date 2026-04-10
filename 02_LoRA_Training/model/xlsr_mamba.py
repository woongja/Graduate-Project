"""
XLSR-Mamba (Bidirectional Mamba SSM) for Audio Deepfake Detection
Source: XLSR-Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
from dataclasses import dataclass, field
from .mamba_blocks import MixerModel


@dataclass
class MambaConfig:
    d_model: int = 64
    n_layer: int = 6
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8


class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        cp_path = '/home/woongjae/wildspoof/xlsr2_300m.pt'
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return emb, layerresult


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + Mamba')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.config = MambaConfig(d_model=args.emb_size, n_layer=args.num_encoders // 2)
        self.conformer = MixerModel(
            d_model=self.config.d_model,
            n_layer=self.config.n_layer,
            ssm_cfg=self.config.ssm_cfg,
            rms_norm=self.config.rms_norm,
            residual_in_fp32=self.config.residual_in_fp32,
            fused_add_norm=self.config.fused_add_norm,
        )

    def forward(self, x):
        x_ssl_feat, _ = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out = self.conformer(x)
        return out
