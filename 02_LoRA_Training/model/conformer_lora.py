"""
Conformer-based Temporal Convolutional Model (TCM) for Audio Deepfake Detection
with LoRA adaptation support

Extracted from tcm_add repository
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from peft import get_peft_model, LoraConfig, TaskType

from .ssl_model import SSLModel


# Helper functions
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def sinusoidal_embedding(n_channels, dim):
    """Sinusoidal positional embedding"""
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)


# Helper classes
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    """
    Head Token Attention
    Reference: https://arxiv.org/pdf/2210.05958.pdf
    """
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0., proj_drop=0.):
        super().__init__()
        self.num_heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.act = nn.GELU()
        self.ht_proj = nn.Linear(dim_head, dim, bias=True)
        self.ht_norm = nn.LayerNorm(dim_head)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_heads, dim))

    def forward(self, x, mask=None):
        B, N, C = x.shape

        # Head token
        head_pos = self.pos_embed.expand(x.shape[0], -1, -1)
        x_ = x.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x_ = x_.mean(dim=2)  # [B, num_heads, dim_head]
        x_ = self.ht_proj(x_).reshape(B, -1, self.num_heads, C // self.num_heads)
        x_ = self.act(self.ht_norm(x_)).flatten(2)
        x_ = x_ + head_pos
        x = torch.cat([x, x_], dim=1)

        # Normal multi-head self-attention
        qkv = self.qkv(x).reshape(B, N + self.num_heads, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N + self.num_heads, C)
        x = self.proj(x)

        # Merge head tokens into cls token
        cls, patch, ht = torch.split(x, [1, N - 1, self.num_heads], dim=1)
        cls = cls + torch.mean(ht, dim=1, keepdim=True) + torch.mean(patch, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)

        x = self.proj_drop(x)

        return x, attn


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    """
    Conformer Block with Feed-Forward, Attention, Convolution modules
    """
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.,
        ff_dropout=0.,
        conv_dropout=0.,
        conv_causal=False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.conv = ConformerConvModule(dim=dim, causal=conv_causal, expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        attn_x, attn_weight = self.attn(x, mask=mask)
        x = attn_x + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x, attn_weight


class MyConformer(nn.Module):
    """Conformer encoder with class token"""
    def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
        super(MyConformer, self).__init__()
        self.dim_head = int(emb_size / heads)
        self.dim = emb_size
        self.heads = heads
        self.kernel_size = kernel_size
        self.n_encoders = n_encoders

        self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)

        self.encoder_blocks = nn.ModuleList([
            ConformerBlock(
                dim=emb_size,
                dim_head=self.dim_head,
                heads=heads,
                ff_mult=ffmult,
                conv_expansion_factor=exp_fac,
                conv_kernel_size=kernel_size
            ) for _ in range(n_encoders)
        ])

        self.class_token = nn.Parameter(torch.rand(1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)

    def forward(self, x, device):
        """
        Args:
            x: [batch, time, emb_size]
            device: torch device
        Returns:
            out: [batch, 2]
            list_attn_weight: list of attention weights
        """
        # Add positional embedding
        x = x + self.positional_emb[:, :x.size(1), :]

        # Add class token
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])  # [batch, 1+time, emb_size]

        list_attn_weight = []
        for layer in self.encoder_blocks:
            x, attn_weight = layer(x)  # [batch, 1+time, emb_size]
            list_attn_weight.append(attn_weight)

        embedding = x[:, 0, :]  # [batch, emb_size] - extract class token
        out = self.fc5(embedding)  # [batch, 2]
        return out, list_attn_weight


class ConformerTCM(nn.Module):
    """
    Conformer-based Temporal Convolutional Model for Audio Deepfake Detection
    Uses XLSR-300M SSL features + Conformer encoder
    """
    def __init__(self, device, ssl_model_path='xlsr2_300m.pt',
                 emb_size=144, num_encoders=4, heads=4, kernel_size=31):
        super().__init__()
        self.device = device

        # SSL model (XLSR-300M)
        self.ssl_model = SSLModel(device, ssl_model_path)
        self.LL = nn.Linear(1024, emb_size)

        print(f'W2V + Conformer (emb_size={emb_size}, encoders={num_encoders})')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.conformer = MyConformer(
            emb_size=emb_size,
            n_encoders=num_encoders,
            heads=heads,
            kernel_size=kernel_size
        )

    def forward(self, x):
        """
        Args:
            x: Raw waveform [batch, length] or [batch, length, 1]
        Returns:
            out: [batch, 2] (bonafide vs spoof)
            attn_score: list of attention weights
        """
        # Extract SSL features
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # [batch, frames, emb_size]

        x = x.unsqueeze(dim=1)  # [batch, 1, frames, emb_size]
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # [batch, frames, emb_size]

        out, attn_score = self.conformer(x, self.device)
        return out, attn_score


def create_conformer_with_lora(device, ssl_model_path='xlsr2_300m.pt',
                                emb_size=144, num_encoders=4, heads=4, kernel_size=31,
                                lora_config=None):
    """
    Create Conformer TCM model with LoRA adaptation

    Args:
        device: torch device
        ssl_model_path: Path to XLSR-300M model
        emb_size: Embedding size for Conformer
        num_encoders: Number of Conformer encoder blocks
        heads: Number of attention heads
        kernel_size: Kernel size for convolutional module
        lora_config: LoRA configuration dict with keys:
            - r: rank (default 8)
            - lora_alpha: scaling factor (default 16)
            - lora_dropout: dropout rate (default 0.0)
            - target_modules: list of module names to apply LoRA

    Returns:
        model: Conformer TCM model with LoRA adapters
    """
    model = ConformerTCM(device, ssl_model_path, emb_size, num_encoders, heads, kernel_size)

    if lora_config is not None:
        # Default LoRA config
        r = lora_config.get('r', 8)
        lora_alpha = lora_config.get('lora_alpha', 16)
        lora_dropout = lora_config.get('lora_dropout', 0.0)
        target_modules = lora_config.get('target_modules', [
            'q_proj', 'v_proj', 'k_proj', 'out_proj',
            'fc1', 'fc2', 'final_proj', 'LL',
            'qkv', 'proj', 'ht_proj'
        ])

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            inference_mode=False
        )

        model = get_peft_model(model, peft_config)
        print(f"LoRA applied to Conformer TCM model")
        model.print_trainable_parameters()

    return model
