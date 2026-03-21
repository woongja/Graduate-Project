# SSAST: Self-Supervised Audio Spectrogram Transformer
# Adapted from https://github.com/YuanGongND/ssast
# Original author: Yuan Gong (MIT)

import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import timm

try:
    from timm.layers import trunc_normal_, to_2tuple
except ImportError:
    from timm.models.layers import trunc_normal_, to_2tuple


# ── timm model name map (old 0.4.5 names → new 1.x names) ──────────────────
_DEIT_MODELS = {
    'tiny':     ('vit_deit_tiny_distilled_patch16_224',  'deit_tiny_distilled_patch16_224'),
    'small':    ('vit_deit_small_distilled_patch16_224', 'deit_small_distilled_patch16_224'),
    'base':     ('vit_deit_base_distilled_patch16_384',  'deit_base_distilled_patch16_384'),
    'base_nokd':('vit_deit_base_patch16_384',            'deit_base_patch16_384'),
}

def _create_deit(model_size, pretrained=False):
    if model_size not in _DEIT_MODELS:
        raise ValueError(f'model_size must be one of {list(_DEIT_MODELS)}')
    old_name, new_name = _DEIT_MODELS[model_size]
    try:
        return timm.create_model(old_name, pretrained=pretrained)
    except Exception:
        return timm.create_model(new_name, pretrained=pretrained)


# ── Custom PatchEmbed: relaxes timm's fixed input-shape constraint ───────────
class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def get_sinusoid_encoding(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    table = np.array([get_position_angle_vec(i) for i in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.FloatTensor(table).unsqueeze(0)


class ASTModel(nn.Module):
    """
    SSAST model for fine-tuning.

    Fine-tuning usage:
        model = ASTModel(
            label_dim=10,
            fshape=16, tshape=16,
            fstride=10, tstride=10,
            input_fdim=128, input_tdim=512,
            model_size='base',          # 'tiny' | 'small' | 'base'
            pretrain_stage=False,
            load_pretrained_mdl_path='pretrained_model/SSAST-Base-Patch-400.pth'
        )
        logits = model(x, task='ft_avgtok')   # x: (B, T, 128)

    Pretrain-stage models are DataParallel-wrapped .pth files.
    The pretrained file must contain keys like 'module.v.patch_embed.proj.weight'.
    """

    def __init__(self, label_dim=527,
                 fshape=16, tshape=16, fstride=16, tstride=16,
                 input_fdim=128, input_tdim=1024,
                 model_size='base',
                 pretrain_stage=True,
                 load_pretrained_mdl_path=None):

        super().__init__()

        # ── Pretraining stage ────────────────────────────────────────────────
        if pretrain_stage:
            if load_pretrained_mdl_path is not None:
                raise ValueError(
                    'load_pretrained_mdl_path must be None during pretraining.')
            if fstride != fshape or tstride != tshape:
                raise ValueError(
                    'fstride must equal fshape and tstride must equal tshape during pretraining.')

            self.v = _create_deit(model_size, pretrained=False)
            _heads_depth = {'tiny': (3, 12), 'small': (6, 12),
                            'base': (12, 12), 'base_nokd': (12, 12)}
            self.heads, self.depth = _heads_depth[model_size]
            self.cls_token_num = 1 if model_size == 'base_nokd' else 2

            self.original_num_patches   = self.v.patch_embed.num_patches
            self.oringal_hw             = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            self.softmax  = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape   = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            self.p_input_fdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False)
            self.p_input_tdim = nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            self.cpredlayer = nn.Sequential(
                nn.Linear(self.original_embedding_dim, self.original_embedding_dim),
                nn.ReLU(),
                nn.Linear(self.original_embedding_dim, 256))
            self.gpredlayer = nn.Sequential(
                nn.Linear(self.original_embedding_dim, self.original_embedding_dim),
                nn.ReLU(),
                nn.Linear(self.original_embedding_dim, 256))
            self.unfold     = nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            nn.init.xavier_normal_(self.mask_embed)

            self.p_f_dim, self.p_t_dim = self.get_shape(
                fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches

            # Replace timm's PatchEmbed (has strict size assertion in timm 1.x)
            # with our custom one that has no assertion.
            new_patch_embed = PatchEmbed(
                img_size=input_fdim, patch_size=fshape,
                in_chans=1, embed_dim=self.original_embedding_dim)
            new_patch_embed.proj = nn.Conv2d(
                1, self.original_embedding_dim,
                kernel_size=(fshape, tshape), stride=(fstride, tstride))
            new_patch_embed.num_patches = num_patches
            self.v.patch_embed = new_patch_embed

            new_pos = nn.Parameter(
                torch.zeros(1, num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos
            trunc_normal_(self.v.pos_embed, std=0.02)

        # ── Fine-tuning stage ────────────────────────────────────────────────
        else:
            if load_pretrained_mdl_path is None:
                raise ValueError('load_pretrained_mdl_path is required for fine-tuning.')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            sd = torch.load(load_pretrained_mdl_path, map_location=device)

            try:
                p_fshape = sd['module.v.patch_embed.proj.weight'].shape[2]
                p_tshape = sd['module.v.patch_embed.proj.weight'].shape[3]
                p_input_fdim = sd['module.p_input_fdim'].item()
                p_input_tdim = sd['module.p_input_tdim'].item()
            except KeyError:
                raise ValueError(
                    'Pretrained model must be saved as a DataParallel state dict '
                    '(keys should start with "module."). '
                    'Wrap with torch.nn.DataParallel before saving.')

            print(f'Loading SSAST pretrained model: {load_pretrained_mdl_path}')

            base_model = ASTModel(
                fshape=p_fshape, tshape=p_tshape,
                fstride=p_fshape, tstride=p_tshape,
                input_fdim=p_input_fdim, input_tdim=p_input_tdim,
                model_size=model_size, pretrain_stage=True)
            base_model = nn.DataParallel(base_model)
            base_model.load_state_dict(sd, strict=False)

            self.v = base_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = base_model.module.cls_token_num

            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(
                fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            p_f_dim = base_model.module.p_f_dim
            p_t_dim = base_model.module.p_t_dim
            num_patches   = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim

            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError(
                    f'Patch shape mismatch: pretrain=({p_fshape},{p_tshape}), '
                    f'finetune=({fshape},{tshape})')

            # Build new proj (with finetuning strides that allow overlap)
            if fstride != p_fshape or tstride != p_tshape:
                new_proj = nn.Conv2d(
                    1, self.original_embedding_dim,
                    kernel_size=(fshape, tshape), stride=(fstride, tstride))
                new_proj.weight = nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            else:
                new_proj = self.v.patch_embed.proj

            # Replace timm's PatchEmbed with our custom one (no size assertion)
            new_patch_embed = PatchEmbed(
                img_size=input_fdim, patch_size=fshape,
                in_chans=1, embed_dim=self.original_embedding_dim)
            new_patch_embed.proj = new_proj
            new_patch_embed.num_patches = num_patches
            self.v.patch_embed = new_patch_embed

            # Positional embedding interpolation
            new_pos = (self.v.pos_embed[:, self.cls_token_num:, :]
                       .detach()
                       .reshape(1, p_num_patches, self.original_embedding_dim)
                       .transpose(1, 2)
                       .reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim))
            if t_dim < p_t_dim:
                new_pos = new_pos[:, :, :, p_t_dim // 2 - t_dim // 2: p_t_dim // 2 - t_dim // 2 + t_dim]
            else:
                new_pos = nn.functional.interpolate(new_pos, size=(p_f_dim, t_dim), mode='bilinear')
            if f_dim < p_f_dim:
                new_pos = new_pos[:, :, p_f_dim // 2 - f_dim // 2: p_f_dim // 2 - f_dim // 2 + f_dim, :]
            else:
                new_pos = nn.functional.interpolate(new_pos, size=(f_dim, t_dim), mode='bilinear')
            new_pos = new_pos.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos], dim=1))

            print(f'  model_size={model_size}, patch=({fshape},{tshape}), '
                  f'stride=({fstride},{tstride}), input=({input_fdim}f x {input_tdim}t), '
                  f'num_patches={num_patches}, label_dim={label_dim}')

    # ────────────────────────────────────────────────────────────────────────
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        dummy = nn.Conv2d(1, self.original_embedding_dim,
                          kernel_size=(fshape, tshape), stride=(fstride, tstride))
        out = dummy(torch.randn(1, 1, input_fdim, input_tdim))
        return out.shape[2], out.shape[3]

    def gen_maskid_patch(self, sequence_len, mask_size, cluster=3):
        mask_id = []
        cur_clus = random.randrange(cluster) + 3
        while len(set(mask_id)) <= mask_size:
            start_id = random.randrange(sequence_len)
            for i in range(cur_clus):
                for j in range(cur_clus):
                    cand = start_id + self.p_t_dim * i + j
                    if 0 < cand < sequence_len:
                        mask_id.append(cand)
        return torch.tensor(list(set(mask_id))[:mask_size])

    def gen_maskid_frame(self, sequence_len, mask_size):
        return torch.tensor(random.sample(range(sequence_len), mask_size))

    # ── Fine-tuning forwards ─────────────────────────────────────────────────
    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls = self.v.cls_token.expand(B, -1, -1)
        if self.cls_token_num == 2:
            dist = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls, dist, x), dim=1)
        else:
            x = torch.cat((cls, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        return self.mlp_head(x)

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls = self.v.cls_token.expand(B, -1, -1)
        if self.cls_token_num == 2:
            dist = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls, dist, x), dim=1)
        else:
            x = torch.cat((cls, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2 if self.cls_token_num == 2 else x[:, 0]
        return self.mlp_head(x)

    # ── Main forward ─────────────────────────────────────────────────────────
    def forward(self, x, task='ft_avgtok', cluster=True, mask_patch=400):
        """
        x    : (batch, time_frames, freq_bins)  e.g. (B, 512, 128)
        task : 'ft_avgtok' | 'ft_cls'  (for finetuning)
        """
        x = x.unsqueeze(1).transpose(2, 3)  # (B, 1, 128, T)

        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        else:
            raise ValueError(f'Unknown task: {task}. Use "ft_avgtok" or "ft_cls" for finetuning.')
