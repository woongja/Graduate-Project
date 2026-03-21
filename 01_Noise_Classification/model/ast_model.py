import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import warnings

# Pretrained model cache dir (01_Noise_Classification/out/pretrained/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['TORCH_HOME'] = os.path.join(_ROOT, 'out', 'pretrained')

if timm.__version__ != '0.4.5':
    warnings.warn(
        f'timm {timm.__version__} detected. AST is tested with timm==0.4.5. '
        'Install with: pip install timm==0.4.5'
    )


class PatchEmbed(nn.Module):
    """Custom PatchEmbed that relaxes timm's fixed input-shape constraint."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ASTModel(nn.Module):
    """
    Audio Spectrogram Transformer (AST).
    Input shape: (batch, time_frames, freq_bins), e.g. (B, 512, 128)

    Args:
        label_dim       : number of output classes
        fstride/tstride : patch stride on freq/time axis (10 = overlap, 16 = no overlap)
        input_fdim      : number of mel bins (default 128)
        input_tdim      : number of time frames (match target_length in dataset)
        imagenet_pretrain: load DeiT ImageNet weights
        audioset_pretrain: load AudioSet-pretrained weights (requires base384 + pretrained pth)
        audioset_pretrain_path: path to audioset pretrained .pth file
        model_size      : 'tiny224' | 'small224' | 'base224' | 'base384'
    """
    def __init__(self, label_dim=11, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=512,
                 imagenet_pretrain=True, audioset_pretrain=False,
                 audioset_pretrain_path=None,
                 model_size='base384', verbose=True):
        super().__init__()

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if not audioset_pretrain:
            _model_map = {
                'tiny224':  'vit_deit_tiny_distilled_patch16_224',
                'small224': 'vit_deit_small_distilled_patch16_224',
                'base224':  'vit_deit_base_distilled_patch16_224',
                'base384':  'vit_deit_base_distilled_patch16_384',
            }
            if model_size not in _model_map:
                raise ValueError(f'model_size must be one of {list(_model_map)}')

            self.v = timm.create_model(_model_map[model_size], pretrained=imagenet_pretrain)
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )

            f_dim, t_dim = self._get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            # 1-channel projection (audio has no colour channels)
            new_proj = nn.Conv2d(1, self.original_embedding_dim,
                                 kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # Positional embedding interpolation
            if imagenet_pretrain:
                new_pos = (self.v.pos_embed[:, 2:, :]
                           .detach()
                           .reshape(1, self.original_num_patches, self.original_embedding_dim)
                           .transpose(1, 2)
                           .reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw))
                new_pos = self._interp_pos(new_pos, f_dim, t_dim)
                new_pos = new_pos.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(
                    torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos], dim=1)
                )
            else:
                new_pos = nn.Parameter(
                    torch.zeros(1, num_patches + 2, self.original_embedding_dim)
                )
                self.v.pos_embed = new_pos
                trunc_normal_(self.v.pos_embed, std=0.02)

        else:
            # AudioSet pretrained
            if not imagenet_pretrain:
                raise ValueError('audioset_pretrain requires imagenet_pretrain=True')
            if model_size != 'base384':
                raise ValueError('Only base384 AudioSet pretrained model is supported')

            pretrain_path = audioset_pretrain_path or os.path.join(
                os.environ['TORCH_HOME'], 'audioset_10_10_0.4593.pth'
            )
            if not os.path.exists(pretrain_path):
                raise FileNotFoundError(
                    f'AudioSet pretrained model not found: {pretrain_path}\n'
                    'Download from: https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            sd = torch.load(pretrain_path, map_location=device)
            base = ASTModel(label_dim=527, fstride=10, tstride=10,
                            input_fdim=128, input_tdim=1024,
                            imagenet_pretrain=False, audioset_pretrain=False,
                            model_size='base384', verbose=False)
            base = nn.DataParallel(base)
            base.load_state_dict(sd, strict=False)
            self.v = base.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )

            f_dim, t_dim = self._get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            new_pos = (self.v.pos_embed[:, 2:, :]
                       .detach()
                       .reshape(1, 1212, 768)
                       .transpose(1, 2)
                       .reshape(1, 768, 12, 101))
            new_pos = self._interp_pos_audioset(new_pos, f_dim, t_dim)
            new_pos = new_pos.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos], dim=1)
            )

        if verbose:
            print('--- AST Model ---')
            print(f'  model_size={model_size}, imagenet_pretrain={imagenet_pretrain}, '
                  f'audioset_pretrain={audioset_pretrain}')
            print(f'  fstride={fstride}, tstride={tstride}, '
                  f'input=({input_fdim}f x {input_tdim}t), label_dim={label_dim}')

    def _get_shape(self, fstride, tstride, input_fdim, input_tdim):
        test = nn.Conv2d(1, self.original_embedding_dim,
                         kernel_size=(16, 16), stride=(fstride, tstride))
        out = test(torch.randn(1, 1, input_fdim, input_tdim))
        return out.shape[2], out.shape[3]

    def _interp_pos(self, pos, f_dim, t_dim):
        hw = self.oringal_hw
        if t_dim <= hw:
            pos = pos[:, :, :, hw // 2 - t_dim // 2: hw // 2 - t_dim // 2 + t_dim]
        else:
            pos = torch.nn.functional.interpolate(pos, size=(hw, t_dim), mode='bilinear')
        if f_dim <= hw:
            pos = pos[:, :, hw // 2 - f_dim // 2: hw // 2 - f_dim // 2 + f_dim, :]
        else:
            pos = torch.nn.functional.interpolate(pos, size=(f_dim, t_dim), mode='bilinear')
        return pos

    def _interp_pos_audioset(self, pos, f_dim, t_dim):
        if t_dim < 101:
            pos = pos[:, :, :, 50 - t_dim // 2: 50 - t_dim // 2 + t_dim]
        else:
            pos = torch.nn.functional.interpolate(pos, size=(12, t_dim), mode='bilinear')
        if f_dim < 12:
            pos = pos[:, :, 6 - f_dim // 2: 6 - f_dim // 2 + f_dim, :]
        elif f_dim > 12:
            pos = torch.nn.functional.interpolate(pos, size=(f_dim, t_dim), mode='bilinear')
        return pos

    @autocast()
    def forward(self, x):
        """
        x: (batch, time_frames, freq_bins)  e.g. (B, 512, 128)
        returns: (batch, label_dim)
        """
        x = x.unsqueeze(1).transpose(2, 3)  # (B, 1, 128, T)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls = self.v.cls_token.expand(B, -1, -1)
        dist = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls, dist, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        return self.mlp_head(x)
