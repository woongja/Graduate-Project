# HTSAT: Hierarchical Token-Semantic Audio Transformer
# Adapted from https://github.com/RetroCirce/HTS-Audio-Transformer
# Original author: Ke Chen (UCSD)

import math
import random
import collections.abc
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def interpolate_seq(x, ratio):
    """Upsample sequence (B, T, C) along T by integer ratio."""
    B, T, C = x.shape
    return x.unsqueeze(2).expand(B, T, ratio, C).contiguous().view(B, T * ratio, C)


def drop_path_fn(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


# ── Building-block layers ─────────────────────────────────────────────────────

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_fn(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features     = out_features     or in_features
        hidden_features  = hidden_features  or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """2D spectrogram → patch token sequence."""
    def __init__(self, img_size=256, patch_size=4, in_chans=1, embed_dim=96,
                 norm_layer=None, patch_stride=(4, 4)):
        super().__init__()
        img_size     = to_2tuple(img_size)
        patch_size   = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size     = img_size
        self.patch_size   = patch_size
        self.patch_stride = patch_stride
        self.grid_size    = (img_size[0] // patch_stride[0],
                             img_size[1] // patch_stride[1])
        self.num_patches  = self.grid_size[0] * self.grid_size[1]
        padding = ((patch_size[0] - patch_stride[0]) // 2,
                   (patch_size[1] - patch_stride[1]) // 2)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input ({H}×{W}) ≠ model ({self.img_size[0]}×{self.img_size[1]})"
        x = self.proj(x).flatten(2).transpose(1, 2)  # B N C
        return self.norm(x)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim        = dim
        self.window_size = window_size
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_f = torch.flatten(coords, 1)
        rel      = coords_f[:, :, None] - coords_f[:, None, :]
        rel      = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj     = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                       + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x)), attn


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 norm_before_mlp='ln'):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size  = shift_size
        if min(input_resolution) <= window_size:
            self.shift_size  = 0
            self.window_size = min(input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path_layer = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if norm_before_mlp == 'ln':
            self.norm2 = nn.LayerNorm(dim)
        else:
            raise NotImplementedError(f'norm_before_mlp={norm_before_mlp}')
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        # SW-MSA mask
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mw = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
            attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(attn_mask == 0, 0.)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W   = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        xw = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        aw, attn = self.attn(xw, mask=self.attn_mask)
        aw = aw.view(-1, self.window_size, self.window_size, C)
        x  = window_reverse(aw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + self.drop_path_layer(x.view(B, H * W, C))
        x = x + self.drop_path_layer(self.mlp(self.norm2(x)))
        return x, attn


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.norm      = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x  = x.view(B, H, W, C)
        x  = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                         x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, norm_before_mlp='ln'):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, norm_before_mlp=norm_before_mlp)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) \
            if downsample is not None else None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
                attn = None
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training and attns:
            attn = torch.mean(torch.cat(attns, 0), 0)
        return x, attn


# ── Main model ────────────────────────────────────────────────────────────────

class HTSATModel(nn.Module):
    """
    Standalone HTSAT fine-tuning wrapper.

    Usage:
        model = HTSATModel(
            num_classes=10,
            load_pretrained_path='pretrained_model/HTSAT-fullset-imagenet-iter1-v2.ckpt'
        )
        logits = model(waveform)   # waveform: (B, num_samples) at 32 kHz
        # logits: (B, num_classes)
    """

    # AudioSet pretrained defaults
    SAMPLE_RATE = 32000
    WINDOW_SIZE = 1024
    HOP_SIZE    = 320
    MEL_BINS    = 64
    FMIN        = 50
    FMAX        = 14000

    # Swin Transformer defaults (HTSAT-tiny)
    SPEC_SIZE    = 256
    PATCH_SIZE   = 4
    PATCH_STRIDE = (4, 4)
    EMBED_DIM    = 96
    DEPTHS       = [2, 2, 6, 2]
    NUM_HEADS    = [4, 8, 16, 32]
    WINDOW_SIZE_SWIN = 8

    def __init__(self,
                 num_classes=10,
                 load_pretrained_path=None,
                 sample_rate=SAMPLE_RATE,
                 window_size=WINDOW_SIZE,
                 hop_size=HOP_SIZE,
                 mel_bins=MEL_BINS,
                 fmin=FMIN,
                 fmax=FMAX,
                 spec_size=SPEC_SIZE,
                 patch_size=PATCH_SIZE,
                 patch_stride=PATCH_STRIDE,
                 embed_dim=EMBED_DIM,
                 depths=None,
                 num_heads=None,
                 window_size_swin=WINDOW_SIZE_SWIN,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_before_mlp='ln',
                 use_checkpoint=False):
        super().__init__()

        depths    = depths    or self.DEPTHS
        num_heads = num_heads or self.NUM_HEADS

        self.num_classes  = num_classes
        self.spec_size    = spec_size
        self.patch_stride = to_2tuple(patch_stride)
        self.mel_bins     = mel_bins
        self.freq_ratio   = spec_size // mel_bins   # 256 // 64 = 4
        self.depths       = depths
        self.num_layers   = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))   # 96*8=768

        # ── Audio front-end ──────────────────────────────────────────────
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
            fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8,  freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(mel_bins)

        # ── Patch embedding ──────────────────────────────────────────────
        self.patch_embed = PatchEmbed(
            img_size=spec_size, patch_size=patch_size,
            in_chans=1, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm, patch_stride=self.patch_stride)
        patches_res = self.patch_embed.grid_size  # (H_p, W_p)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # ── Swin Transformer stages ──────────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(patches_res[0] // (2 ** i),
                                   patches_res[1] // (2 ** i)),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size_swin,
                mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                norm_before_mlp=norm_before_mlp)
            self.layers.append(layer)

        self.norm    = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # ── Token-semantic head ──────────────────────────────────────────
        # SF: freq bins after all downsampling
        SF = spec_size // (2 ** (self.num_layers - 1)) // self.patch_stride[0] // self.freq_ratio
        self.tscam_conv = nn.Conv2d(
            in_channels=self.num_features, out_channels=num_classes,
            kernel_size=(SF, 3), padding=(0, 1))
        self.head = nn.Linear(num_classes, num_classes)

        self.apply(self._init_weights)

        if load_pretrained_path is not None:
            self._load_pretrained(load_pretrained_path)

    # ─────────────────────────────────────────────────────────────────────────

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,   0)
            nn.init.constant_(m.weight, 1.0)

    def _load_pretrained(self, ckpt_path):
        """Load AudioSet Lightning checkpoint, drop mismatched head weights."""
        print(f'Loading HTSAT pretrained: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')

        sd = ckpt.get('state_dict', ckpt)  # Lightning wraps in 'state_dict'

        # Strip 'sed_model.' prefix added by Lightning wrapper
        sd = {(k[len('sed_model.'):] if k.startswith('sed_model.') else k): v
              for k, v in sd.items()}

        # Drop head & tscam_conv (pretrained for 527 AudioSet classes)
        sd = {k: v for k, v in sd.items()
              if not k.startswith('head') and not k.startswith('tscam_conv')}

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f'  Loaded  — missing: {len(missing)}, unexpected: {len(unexpected)}')

    # ── Spectrogram geometry helpers ──────────────────────────────────────────

    def _reshape_wav2img(self, x):
        """Reshape (B,1,T,F) mel spec into (B,1,spec_size,spec_size) for Swin."""
        B, C, T, Fq = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        if T < target_T:
            x = F.interpolate(x, (target_T, x.shape[3]), mode='bicubic', align_corners=True)
        if Fq < target_F:
            x = F.interpolate(x, (x.shape[2], target_F), mode='bicubic', align_corners=True)
        # tile frequency into rows, time into columns
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(B, C, x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, C, x.shape[2] * x.shape[3], x.shape[4])
        return x

    def _crop_wav(self, x, crop_size, spe_pos=None):
        T  = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3], device=x.device)
        for i in range(len(x)):
            pos = random.randint(0, T - crop_size - 1) if spe_pos is None else spe_pos
            tx[i][0] = x[i, 0, pos:pos + crop_size, :]
        return tx

    # ── Feature extraction ────────────────────────────────────────────────────

    def _forward_swin(self, x):
        """(B,1,spec_size,spec_size) → (B, num_classes) clip logits."""
        frames_num = x.shape[2]
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm(x)

        B, N, C = x.shape
        SF = frames_num // (2 ** (self.num_layers - 1)) // self.patch_stride[0]
        ST = frames_num // (2 ** (self.num_layers - 1)) // self.patch_stride[1]
        x  = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)

        c_freq_bin = SF // self.freq_ratio
        x = x.reshape(B, C, SF // c_freq_bin, c_freq_bin, ST)
        x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)

        x = self.tscam_conv(x)          # B, num_classes, 1, T'
        x = torch.flatten(x, 2)         # B, num_classes, T'
        x = self.avgpool(x).squeeze(-1) # B, num_classes
        x = self.head(x)                # B, num_classes
        return x

    # ── Public forward ────────────────────────────────────────────────────────

    def forward(self, waveform):
        """
        Args:
            waveform : (B, num_samples) — raw audio at 32 kHz
        Returns:
            logits   : (B, num_classes) — raw class scores
        """
        x = self.spectrogram_extractor(waveform)   # (B, 1, T, freq_bins)
        x = self.logmel_extractor(x)               # (B, 1, T, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        target_T = int(self.spec_size * self.freq_ratio)

        if x.shape[2] > target_T:
            if self.training:
                x = self._crop_wav(x, crop_size=target_T)
                x = self._reshape_wav2img(x)
                return self._forward_swin(x)
            else:
                # sliding-window average for long audio
                overlap  = (x.shape[2] - 1) // 4
                crop_sz  = (x.shape[2] - 1) // 2
                clips    = []
                for pos in range(0, x.shape[2] - crop_sz - 1, overlap):
                    tx = self._crop_wav(x, crop_size=crop_sz, spe_pos=pos)
                    tx = self._reshape_wav2img(tx)
                    clips.append(self._forward_swin(tx))
                return torch.stack(clips).mean(0)
        else:
            x = self._reshape_wav2img(x)
            return self._forward_swin(x)
