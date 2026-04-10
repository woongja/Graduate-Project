"""
CNN8RNN 3-Feature Fusion with Softmax Attention
Features compete for importance (sum of weights = 1)
"""
import torch
import torch.nn as nn
from model.cnn8rnn_3ff_base import SpectrogramBranch, MFCCBranch, F0Branch


class SoftmaxAttention(nn.Module):
    """Softmax-based competitive attention mechanism"""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, spec_feat, mfcc_feat, f0_feat):
        concat_features = torch.cat([spec_feat, mfcc_feat, f0_feat], dim=1)
        logits = self.attention(concat_features)
        weights = torch.softmax(logits / self.temperature, dim=1)  # [B, 3], sum=1

        w_spec = weights[:, 0:1]
        w_mfcc = weights[:, 1:2]
        w_f0 = weights[:, 2:3]

        weighted_spec = spec_feat * w_spec
        weighted_mfcc = mfcc_feat * w_mfcc
        weighted_f0 = f0_feat * w_f0

        weighted_features = torch.cat([weighted_spec, weighted_mfcc, weighted_f0], dim=1)

        return weighted_features, weights


class CNN8RNN_3FF_Softmax(nn.Module):
    """CNN8RNN 3FF with Softmax Competitive Attention"""

    def __init__(
        self,
        num_classes: int,
        branch_output_dim: int = 512,
        dropout: float = 0.5
    ):
        super().__init__()

        self.spec_branch = SpectrogramBranch(output_dim=branch_output_dim)
        self.mfcc_branch = MFCCBranch(output_dim=branch_output_dim)
        self.f0_branch = F0Branch(output_dim=branch_output_dim)

        self.feature_attention = SoftmaxAttention(branch_output_dim)

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

        fused, self.attention_weights = self.feature_attention(spec_feat, mfcc_feat, f0_feat)

        logits = self.classifier(fused)

        return logits

    def get_attention_weights(self):
        return self.attention_weights if hasattr(self, 'attention_weights') else None


def create_cnn8rnn_3ff_softmax(num_classes: int, branch_output_dim: int = 512, dropout: float = 0.5):
    return CNN8RNN_3FF_Softmax(num_classes=num_classes, branch_output_dim=branch_output_dim, dropout=dropout)
