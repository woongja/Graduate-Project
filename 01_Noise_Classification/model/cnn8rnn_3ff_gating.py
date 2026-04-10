"""
CNN8RNN 3-Feature Fusion with Gating Mechanism
Element-wise gating based on global context
"""
import torch
import torch.nn as nn
from model.cnn8rnn_3ff_base import SpectrogramBranch, MFCCBranch, F0Branch


class FeatureGating(nn.Module):
    """Gating mechanism for adaptive feature fusion"""
    def __init__(self, feature_dim: int):
        super().__init__()

        # Global context encoder
        self.global_context = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Feature-specific gate generators
        self.spec_gate = nn.Sequential(nn.Linear(256, feature_dim), nn.Sigmoid())
        self.mfcc_gate = nn.Sequential(nn.Linear(256, feature_dim), nn.Sigmoid())
        self.f0_gate = nn.Sequential(nn.Linear(256, feature_dim), nn.Sigmoid())

    def forward(self, spec_feat, mfcc_feat, f0_feat):
        concat_features = torch.cat([spec_feat, mfcc_feat, f0_feat], dim=1)
        context = self.global_context(concat_features)

        gate_spec = self.spec_gate(context)
        gate_mfcc = self.mfcc_gate(context)
        gate_f0 = self.f0_gate(context)

        gated_spec = spec_feat * gate_spec
        gated_mfcc = mfcc_feat * gate_mfcc
        gated_f0 = f0_feat * gate_f0

        gated_features = torch.cat([gated_spec, gated_mfcc, gated_f0], dim=1)

        gate_stats = {
            'spec_mean': gate_spec.mean().item(),
            'mfcc_mean': gate_mfcc.mean().item(),
            'f0_mean': gate_f0.mean().item()
        }

        return gated_features, gate_stats


class CNN8RNN_3FF_Gating(nn.Module):
    """CNN8RNN 3FF with Gating Mechanism"""

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

        self.feature_gating = FeatureGating(branch_output_dim)

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

        fused, self.gate_stats = self.feature_gating(spec_feat, mfcc_feat, f0_feat)

        logits = self.classifier(fused)

        return logits

    def get_gate_stats(self):
        return self.gate_stats if hasattr(self, 'gate_stats') else None


def create_cnn8rnn_3ff_gating(num_classes: int, branch_output_dim: int = 512, dropout: float = 0.5):
    return CNN8RNN_3FF_Gating(num_classes=num_classes, branch_output_dim=branch_output_dim, dropout=dropout)
