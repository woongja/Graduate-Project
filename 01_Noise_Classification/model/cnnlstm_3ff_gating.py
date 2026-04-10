"""
CNN+LSTM 3-Feature Fusion (3FF) Model with Gating Mechanism
Fuses Spectrogram + MFCC + F0 features using element-wise gating
"""
import torch
import torch.nn as nn


class SpectrogramBranch(nn.Module):
    """CNN+LSTM branch for Spectrogram feature extraction"""

    def __init__(self, in_channels=1, output_dim=512, lstm_hidden=128, lstm_layers=2):
        super().__init__()

        # CNN layers for spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0.0
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(2)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out.mean(dim=1)
        x = self.fc(x)
        return x


class MFCCBranch(nn.Module):
    """CNN+LSTM branch for MFCC feature extraction"""

    def __init__(self, in_channels=1, output_dim=512, lstm_hidden=64, lstm_layers=2):
        super().__init__()

        # CNN layers for MFCC
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0.0
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(2)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out.mean(dim=1)
        x = self.fc(x)
        return x


class F0Branch(nn.Module):
    """CNN+LSTM branch for F0 (fundamental frequency) feature extraction"""

    def __init__(self, output_dim=512, lstm_hidden=64, lstm_layers=2):
        super().__init__()

        # 1D CNN for F0 contour
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0.0
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out.mean(dim=1)
        x = self.fc(x)
        return x


class FeatureGating(nn.Module):
    """
    Gating mechanism for adaptive feature fusion
    Uses global context to generate element-wise gates for each feature
    """
    def __init__(self, feature_dim: int):
        super().__init__()

        # Global context encoder
        self.global_context = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU()
        )

        # Feature-specific gate generators (element-wise)
        self.spec_gate = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.Sigmoid()
        )

        self.mfcc_gate = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.Sigmoid()
        )

        self.f0_gate = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, spec_feat, mfcc_feat, f0_feat):
        """
        Args:
            spec_feat: [batch, feature_dim]
            mfcc_feat: [batch, feature_dim]
            f0_feat: [batch, feature_dim]
        Returns:
            gated_features: [batch, feature_dim*3] - Gated concatenated features
            gate_stats: dict - Statistics of gate activations
        """
        # Extract global context from all features
        concat_features = torch.cat([spec_feat, mfcc_feat, f0_feat], dim=1)
        context = self.global_context(concat_features)  # [B, 256]

        # Generate feature-specific gates
        gate_spec = self.spec_gate(context)  # [B, feature_dim]
        gate_mfcc = self.mfcc_gate(context)  # [B, feature_dim]
        gate_f0 = self.f0_gate(context)      # [B, feature_dim]

        # Apply gates (element-wise multiplication)
        gated_spec = spec_feat * gate_spec
        gated_mfcc = mfcc_feat * gate_mfcc
        gated_f0 = f0_feat * gate_f0

        # Concatenate gated features
        gated_features = torch.cat([gated_spec, gated_mfcc, gated_f0], dim=1)

        # Store gate statistics for analysis
        gate_stats = {
            'spec_mean': gate_spec.mean().item(),
            'mfcc_mean': gate_mfcc.mean().item(),
            'f0_mean': gate_f0.mean().item()
        }

        return gated_features, gate_stats


class CNNLSTM_3FF_Gating(nn.Module):
    """
    3-Feature Fusion Network with Gating Mechanism

    Key improvement: Uses element-wise gating based on global context,
    allowing fine-grained control over each feature dimension
    """

    def __init__(
        self,
        num_classes: int,
        branch_output_dim: int = 512,
        spec_lstm_hidden: int = 128,
        spec_lstm_layers: int = 2,
        mfcc_lstm_hidden: int = 64,
        mfcc_lstm_layers: int = 2,
        f0_lstm_hidden: int = 64,
        f0_lstm_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()

        # Feature extraction branches
        self.spec_branch = SpectrogramBranch(
            in_channels=1,
            output_dim=branch_output_dim,
            lstm_hidden=spec_lstm_hidden,
            lstm_layers=spec_lstm_layers
        )

        self.mfcc_branch = MFCCBranch(
            in_channels=1,
            output_dim=branch_output_dim,
            lstm_hidden=mfcc_lstm_hidden,
            lstm_layers=mfcc_lstm_layers
        )

        self.f0_branch = F0Branch(
            output_dim=branch_output_dim,
            lstm_hidden=f0_lstm_hidden,
            lstm_layers=f0_lstm_layers
        )

        # Feature gating mechanism
        self.feature_gating = FeatureGating(branch_output_dim)

        # Fusion dimensions
        fused_dim = branch_output_dim * 3  # 1536

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        self.num_classes = num_classes
        self.branch_output_dim = branch_output_dim

    def forward(self, spec, mfcc, f0):
        # Extract features from each branch
        spec_feat = self.spec_branch(spec)
        mfcc_feat = self.mfcc_branch(mfcc)
        f0_feat = self.f0_branch(f0)

        # Apply gating mechanism
        fused, self.gate_stats = self.feature_gating(spec_feat, mfcc_feat, f0_feat)

        # Classification
        logits = self.classifier(fused)

        return logits

    def get_gate_stats(self):
        """Get the last computed gate statistics"""
        return self.gate_stats if hasattr(self, 'gate_stats') else None


def create_cnnlstm_3ff_gating(
    num_classes: int,
    branch_output_dim: int = 512,
    dropout: float = 0.5
):
    return CNNLSTM_3FF_Gating(
        num_classes=num_classes,
        branch_output_dim=branch_output_dim,
        dropout=dropout
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_cnnlstm_3ff_gating(num_classes=10).to(device)

    spec = torch.randn(4, 1, 128, 126).to(device)
    mfcc = torch.randn(4, 1, 40, 126).to(device)
    f0 = torch.randn(4, 1, 126).to(device)

    output = model(spec, mfcc, f0)
    stats = model.get_gate_stats()

    print("=" * 60)
    print("CNN+LSTM 3FF with Gating Mechanism")
    print("=" * 60)
    print(f"Output: {output.shape}")
    print(f"Gate statistics:")
    print(f"  Spec mean: {stats['spec_mean']:.3f}")
    print(f"  MFCC mean: {stats['mfcc_mean']:.3f}")
    print(f"  F0 mean:   {stats['f0_mean']:.3f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
