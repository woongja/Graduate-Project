"""
CNN+LSTM 3-Feature Fusion (3FF) Model with Softmax Attention
Fuses Spectrogram + MFCC + F0 features using competitive softmax weights
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
        x = self.cnn(x)              # [B, 256, F', T']
        x = x.mean(2)                # [B, 256, T']
        x = x.permute(0, 2, 1)       # [B, T', 256]
        lstm_out, _ = self.lstm(x)   # [B, T', lstm_hidden*2]
        x = lstm_out.mean(dim=1)     # [B, lstm_hidden*2]
        x = self.fc(x)               # [B, output_dim]
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
        x = self.cnn(x)              # [B, 128, F', T']
        x = x.mean(2)                # [B, 128, T']
        x = x.permute(0, 2, 1)       # [B, T', 128]
        lstm_out, _ = self.lstm(x)   # [B, T', lstm_hidden*2]
        x = lstm_out.mean(dim=1)     # [B, lstm_hidden*2]
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
        x = self.conv(x)             # [B, 128, T']
        x = x.permute(0, 2, 1)       # [B, T', 128]
        lstm_out, _ = self.lstm(x)   # [B, T', lstm_hidden*2]
        x = lstm_out.mean(dim=1)     # [B, lstm_hidden*2]
        x = self.fc(x)               # [B, output_dim]
        return x


class SoftmaxAttention(nn.Module):
    """
    Softmax-based competitive attention mechanism
    Forces features to compete for importance (sum of weights = 1)
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        # Attention weight generator for 3 features
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 3, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
            # No activation here - softmax applied in forward
        )

        # Temperature parameter for controlling attention sharpness
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, spec_feat, mfcc_feat, f0_feat):
        """
        Args:
            spec_feat: [batch, feature_dim]
            mfcc_feat: [batch, feature_dim]
            f0_feat: [batch, feature_dim]
        Returns:
            weighted_features: [batch, feature_dim*3] - Weighted concatenated features
            weights: [batch, 3] - Softmax attention weights (sum=1)
        """
        # Concatenate all features
        concat_features = torch.cat([spec_feat, mfcc_feat, f0_feat], dim=1)  # [B, feature_dim*3]

        # Compute attention logits
        logits = self.attention(concat_features)  # [B, 3]

        # Apply temperature-scaled softmax
        weights = torch.softmax(logits / self.temperature, dim=1)  # [B, 3], sum=1

        # Apply weights to each feature
        w_spec = weights[:, 0:1]  # [B, 1]
        w_mfcc = weights[:, 1:2]  # [B, 1]
        w_f0 = weights[:, 2:3]    # [B, 1]

        weighted_spec = spec_feat * w_spec    # [B, feature_dim]
        weighted_mfcc = mfcc_feat * w_mfcc    # [B, feature_dim]
        weighted_f0 = f0_feat * w_f0          # [B, feature_dim]

        # Concatenate weighted features
        weighted_features = torch.cat([weighted_spec, weighted_mfcc, weighted_f0], dim=1)

        return weighted_features, weights


class CNNLSTM_3FF_Softmax(nn.Module):
    """
    3-Feature Fusion Network with Softmax Competitive Attention

    Key improvement: Uses softmax instead of sigmoid for attention weights,
    forcing features to compete (sum of weights = 1)
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

        # Softmax attention mechanism
        self.feature_attention = SoftmaxAttention(branch_output_dim)

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

        # Apply softmax attention
        fused, self.attention_weights = self.feature_attention(spec_feat, mfcc_feat, f0_feat)

        # Classification
        logits = self.classifier(fused)

        return logits

    def get_attention_weights(self):
        """Get the last computed attention weights"""
        return self.attention_weights if hasattr(self, 'attention_weights') else None


def create_cnnlstm_3ff_softmax(
    num_classes: int,
    branch_output_dim: int = 512,
    dropout: float = 0.5
):
    return CNNLSTM_3FF_Softmax(
        num_classes=num_classes,
        branch_output_dim=branch_output_dim,
        dropout=dropout
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_cnnlstm_3ff_softmax(num_classes=10).to(device)

    spec = torch.randn(4, 1, 128, 126).to(device)
    mfcc = torch.randn(4, 1, 40, 126).to(device)
    f0 = torch.randn(4, 1, 126).to(device)

    output = model(spec, mfcc, f0)
    weights = model.get_attention_weights()

    print("=" * 60)
    print("CNN+LSTM 3FF with Softmax Attention")
    print("=" * 60)
    print(f"Output: {output.shape}")
    print(f"Attention weights: {weights.shape}")
    print(f"Weights sum: {weights.sum(dim=1)[0]:.4f} (should be 1.0)")
    print(f"Sample weights: spec={weights[0,0]:.3f}, mfcc={weights[0,1]:.3f}, f0={weights[0,2]:.3f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
