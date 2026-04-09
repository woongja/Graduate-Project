"""
CNN+LSTM 3-Feature Fusion (3FF) Model with Feature-Level Attention
Fuses Spectrogram + MFCC + F0 features using learned attention weights
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
        """
        Args:
            x: [batch, 1, freq_bins, time_frames] - Spectrogram input
        Returns:
            features: [batch, output_dim] - Extracted features
        """
        x = self.cnn(x)              # [B, 256, F', T']
        x = x.mean(2)                # [B, 256, T'] - frequency pooling
        x = x.permute(0, 2, 1)       # [B, T', 256]
        lstm_out, _ = self.lstm(x)   # [B, T', lstm_hidden*2]
        x = lstm_out.mean(dim=1)     # [B, lstm_hidden*2] - temporal pooling
        x = self.fc(x)               # [B, output_dim]
        return x


class MFCCBranch(nn.Module):
    """CNN+LSTM branch for MFCC feature extraction"""

    def __init__(self, in_channels=1, output_dim=512, lstm_hidden=64, lstm_layers=2):
        super().__init__()

        # CNN layers for MFCC (smaller kernel for lower freq resolution)
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
        """
        Args:
            x: [batch, 1, mfcc_dims, time_frames] - MFCC input
        Returns:
            features: [batch, output_dim] - Extracted features
        """
        x = self.cnn(x)              # [B, 128, F', T']
        x = x.mean(2)                # [B, 128, T'] - frequency pooling
        x = x.permute(0, 2, 1)       # [B, T', 128]
        lstm_out, _ = self.lstm(x)   # [B, T', lstm_hidden*2]
        x = lstm_out.mean(dim=1)     # [B, lstm_hidden*2] - temporal pooling
        x = self.fc(x)               # [B, output_dim]
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
        """
        Args:
            x: [batch, 1, time_frames] - F0 contour input
        Returns:
            features: [batch, output_dim] - Extracted features
        """
        x = self.conv(x)             # [B, 128, T']
        x = x.permute(0, 2, 1)       # [B, T', 128]
        lstm_out, _ = self.lstm(x)   # [B, T', lstm_hidden*2]
        x = lstm_out.mean(dim=1)     # [B, lstm_hidden*2] - temporal pooling
        x = self.fc(x)               # [B, output_dim]
        return x


class FeatureAttention(nn.Module):
    """
    Feature-level attention mechanism
    Learns adaptive weights for each feature type
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        # Attention weight generator for 3 features
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 3, 256),
            nn.GELU(),
            nn.Linear(256, 3),
            nn.Sigmoid()  # Output attention weights for [spec, mfcc, f0]
        )

    def forward(self, spec_feat, mfcc_feat, f0_feat):
        """
        Args:
            spec_feat: [batch, feature_dim]
            mfcc_feat: [batch, feature_dim]
            f0_feat: [batch, feature_dim]
        Returns:
            weighted_features: [batch, feature_dim*3] - Weighted concatenated features
            weights: [batch, 3] - Attention weights for each feature
        """
        # Concatenate all features
        concat_features = torch.cat([spec_feat, mfcc_feat, f0_feat], dim=1)  # [B, feature_dim*3]

        # Compute attention weights
        weights = self.attention(concat_features)  # [B, 3]

        # Apply weights to each feature
        w_spec = weights[:, 0:1]  # [B, 1]
        w_mfcc = weights[:, 1:2]  # [B, 1]
        w_f0 = weights[:, 2:3]    # [B, 1]

        weighted_spec = spec_feat * w_spec    # [B, feature_dim]
        weighted_mfcc = mfcc_feat * w_mfcc    # [B, feature_dim]
        weighted_f0 = f0_feat * w_f0          # [B, feature_dim]

        # Concatenate weighted features
        weighted_features = torch.cat([weighted_spec, weighted_mfcc, weighted_f0], dim=1)  # [B, feature_dim*3]

        return weighted_features, weights


class CNNLSTM_3FF_Weight(nn.Module):
    """
    3-Feature Fusion Network with Feature-Level Attention: Spectrogram + MFCC + F0

    Architecture:
    - Spectrogram Branch: CNN+LSTM for spectral features
    - MFCC Branch: CNN+LSTM for cepstral coefficients
    - F0 Branch: CNN+LSTM for fundamental frequency features
    - Fusion: Adaptive weighted fusion with learned attention + MLP classifier
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
        """
        Args:
            num_classes: Number of output classes
            branch_output_dim: Output dimension of each branch
            spec_lstm_hidden: LSTM hidden size for spectrogram branch
            spec_lstm_layers: Number of LSTM layers for spectrogram branch
            mfcc_lstm_hidden: LSTM hidden size for MFCC branch
            mfcc_lstm_layers: Number of LSTM layers for MFCC branch
            f0_lstm_hidden: LSTM hidden size for F0 branch
            f0_lstm_layers: Number of LSTM layers for F0 branch
            dropout: Dropout rate for classifier
        """
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

        # Feature attention mechanism
        self.feature_attention = FeatureAttention(branch_output_dim)

        # Fusion dimensions
        fused_dim = branch_output_dim * 3  # 1536

        # Classifier: 1536 → 1024 → 512 → 10
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

        # Store config
        self.num_classes = num_classes
        self.branch_output_dim = branch_output_dim

    def forward(self, spec, mfcc, f0):
        """
        Args:
            spec: [batch, 1, freq_bins, time_frames] - Mel Spectrogram
            mfcc: [batch, 1, mfcc_dims, time_frames] - MFCC
            f0: [batch, 1, time_frames] - F0 contour
        Returns:
            logits: [batch, num_classes] - Classification logits
        """
        # Extract features from each branch
        spec_feat = self.spec_branch(spec)  # [B, branch_output_dim]
        mfcc_feat = self.mfcc_branch(mfcc)  # [B, branch_output_dim]
        f0_feat = self.f0_branch(f0)        # [B, branch_output_dim]

        # Apply feature-level attention
        fused, self.attention_weights = self.feature_attention(spec_feat, mfcc_feat, f0_feat)  # [B, 1536], [B, 3]

        # Classification
        logits = self.classifier(fused)  # [B, num_classes]

        return logits

    def get_features(self, spec, mfcc, f0):
        """
        Extract weighted fused features without classification

        Args:
            spec: [batch, 1, freq_bins, time_frames]
            mfcc: [batch, 1, mfcc_dims, time_frames]
            f0: [batch, 1, time_frames]
        Returns:
            features: [batch, branch_output_dim*3] - Weighted fused features
            weights: [batch, 3] - Attention weights
        """
        spec_feat = self.spec_branch(spec)
        mfcc_feat = self.mfcc_branch(mfcc)
        f0_feat = self.f0_branch(f0)
        fused, weights = self.feature_attention(spec_feat, mfcc_feat, f0_feat)
        return fused, weights

    def get_attention_weights(self):
        """
        Get the last computed attention weights

        Returns:
            weights: [batch, 3] - Attention weights for [spec, mfcc, f0]
        """
        return self.attention_weights if hasattr(self, 'attention_weights') else None


# Factory function for easy model creation
def create_cnnlstm_3ff_weight(
    num_classes: int,
    branch_output_dim: int = 512,
    dropout: float = 0.5
) -> CNNLSTM_3FF_Weight:
    """
    Create a CNN+LSTM 3-Feature Fusion model with Feature-Level Attention

    Args:
        num_classes: Number of output classes
        branch_output_dim: Output dimension of each feature branch
        dropout: Dropout rate

    Returns:
        model: CNNLSTM_3FF_Weight model instance
    """
    return CNNLSTM_3FF_Weight(
        num_classes=num_classes,
        branch_output_dim=branch_output_dim,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    num_classes = 10  # 10 noise types
    batch_size = 4

    # Create model
    model = create_cnnlstm_3ff_weight(
        num_classes=num_classes,
        branch_output_dim=512,
        dropout=0.5
    ).to(device)

    # Test inputs
    spec = torch.randn(batch_size, 1, 128, 126).to(device)  # Mel-spectrogram
    mfcc = torch.randn(batch_size, 1, 40, 126).to(device)   # MFCC (40 coefficients)
    f0 = torch.randn(batch_size, 1, 126).to(device)         # F0 contour

    # Forward pass
    output = model(spec, mfcc, f0)
    features, weights = model.get_features(spec, mfcc, f0)

    print("=" * 60)
    print("CNN+LSTM 3FF with Feature-Level Attention Model Test")
    print("=" * 60)
    print(f"Input shapes:")
    print(f"  Spectrogram: {spec.shape}")
    print(f"  MFCC:        {mfcc.shape}")
    print(f"  F0:          {f0.shape}")
    print(f"\nOutput shapes:")
    print(f"  Logits:      {output.shape}")
    print(f"  Features:    {features.shape}")
    print(f"  Weights:     {weights.shape}")
    print(f"\nAttention weights (example):")
    print(f"  Batch 0: spec={weights[0,0]:.3f}, mfcc={weights[0,1]:.3f}, f0={weights[0,2]:.3f}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    # Test with different input sizes
    print("\nTesting with different time dimensions:")
    for time_frames in [100, 126, 200]:
        spec_test = torch.randn(2, 1, 128, time_frames).to(device)
        mfcc_test = torch.randn(2, 1, 40, time_frames).to(device)
        f0_test = torch.randn(2, 1, time_frames).to(device)
        output_test = model(spec_test, mfcc_test, f0_test)
        print(f"  Time={time_frames}: Output shape = {output_test.shape}")
