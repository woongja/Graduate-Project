"""
CNN8RNN 3-Feature Fusion with Cross-Modal Attention
Features interact through multi-head self-attention
"""
import torch
import torch.nn as nn
import math
from model.cnn8rnn_3ff_base import SpectrogramBranch, MFCCBranch, F0Branch


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention mechanism"""
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, spec_feat, mfcc_feat, f0_feat):
        batch_size = spec_feat.size(0)

        # Stack features: [B, 3, feature_dim]
        x = torch.stack([spec_feat, mfcc_feat, f0_feat], dim=1)
        residual = x

        # Multi-head attention
        Q = self.query(x).view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 3, self.feature_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection + Layer norm
        output = self.layer_norm(residual + self.dropout(attn_output))

        # Flatten to [B, feature_dim*3]
        output_flat = output.reshape(batch_size, -1)

        return output_flat, attn_weights


class CNN8RNN_3FF_CrossModal(nn.Module):
    """CNN8RNN 3FF with Cross-Modal Attention"""

    def __init__(
        self,
        num_classes: int,
        branch_output_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.5
    ):
        super().__init__()

        self.spec_branch = SpectrogramBranch(output_dim=branch_output_dim)
        self.mfcc_branch = MFCCBranch(output_dim=branch_output_dim)
        self.f0_branch = F0Branch(output_dim=branch_output_dim)

        self.cross_modal_attention = CrossModalAttention(
            feature_dim=branch_output_dim,
            num_heads=num_attention_heads,
            dropout=0.1
        )

        fused_dim = branch_output_dim * 3

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        self.num_classes = num_classes
        self.branch_output_dim = branch_output_dim

    def forward(self, spec, mfcc, f0):
        spec_feat = self.spec_branch(spec)
        mfcc_feat = self.mfcc_branch(mfcc)
        f0_feat = self.f0_branch(f0)

        fused, self.attention_weights = self.cross_modal_attention(spec_feat, mfcc_feat, f0_feat)

        logits = self.classifier(fused)

        return logits

    def get_attention_weights(self):
        return self.attention_weights if hasattr(self, 'attention_weights') else None


def create_cnn8rnn_3ff_crossmodal(num_classes: int, branch_output_dim: int = 512,
                                   num_attention_heads: int = 8, dropout: float = 0.5):
    return CNN8RNN_3FF_CrossModal(
        num_classes=num_classes,
        branch_output_dim=branch_output_dim,
        num_attention_heads=num_attention_heads,
        dropout=dropout
    )
