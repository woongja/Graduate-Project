"""
CNN+LSTM 2-Feature Fusion (2FF) Model
Fuses Spectrogram and F0 features for audio classification
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
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 1, freq_bins, time_frames] - Spectrogram input
        Returns:
            features: [batch, output_dim] - Extracted features
        """
        # CNN feature extraction
        x = self.cnn(x)  # [B, 256, F', T']

        # Frequency pooling
        x = x.mean(2)  # [B, 256, T']

        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # [B, T', 256]

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)  # [B, T', lstm_hidden*2]

        # Temporal pooling
        x = lstm_out.mean(dim=1)  # [B, lstm_hidden*2]

        # Final projection
        x = self.fc(x)  # [B, output_dim]

        return x


class F0Branch(nn.Module):
    """CNN+LSTM branch for F0 (fundamental frequency) feature extraction"""

    def __init__(self, output_dim=512, lstm_hidden=64, lstm_layers=2):
        super().__init__()

        # 1D CNN for F0 contour
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 1, time_frames] - F0 contour input
        Returns:
            features: [batch, output_dim] - Extracted features
        """
        # 1D CNN feature extraction
        x = self.conv(x)  # [B, 128, T']

        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # [B, T', 128]

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)  # [B, T', lstm_hidden*2]

        # Temporal pooling
        x = lstm_out.mean(dim=1)  # [B, lstm_hidden*2]

        # Final projection
        x = self.fc(x)  # [B, output_dim]

        return x


class CNNLSTM_2FF(nn.Module):
    """
    2-Feature Fusion Network: Spectrogram + F0

    Architecture:
    - Spectrogram Branch: CNN+LSTM for spectral features
    - F0 Branch: CNN+LSTM for fundamental frequency features
    - Fusion: Concatenation + MLP classifier
    """

    def __init__(
        self,
        num_classes: int,
        branch_output_dim: int = 512,
        fusion_hidden_dim: int = 512,
        spec_lstm_hidden: int = 128,
        spec_lstm_layers: int = 2,
        f0_lstm_hidden: int = 64,
        f0_lstm_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes: Number of output classes
            branch_output_dim: Output dimension of each branch
            fusion_hidden_dim: Hidden dimension of fusion layer
            spec_lstm_hidden: LSTM hidden size for spectrogram branch
            spec_lstm_layers: Number of LSTM layers for spectrogram branch
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

        self.f0_branch = F0Branch(
            output_dim=branch_output_dim,
            lstm_hidden=f0_lstm_hidden,
            lstm_layers=f0_lstm_layers
        )

        # Fusion classifier
        fused_dim = branch_output_dim * 2  # Concatenate 2 branches

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

        # Store config
        self.num_classes = num_classes
        self.branch_output_dim = branch_output_dim

    def forward(self, spec, f0):
        """
        Args:
            spec: [batch, 1, freq_bins, time_frames] - Spectrogram
            f0: [batch, 1, time_frames] - F0 contour
        Returns:
            logits: [batch, num_classes] - Classification logits
        """
        # Extract features from each branch
        spec_feat = self.spec_branch(spec)  # [B, branch_output_dim]
        f0_feat = self.f0_branch(f0)        # [B, branch_output_dim]

        # Feature fusion (concatenation)
        fused = torch.cat([spec_feat, f0_feat], dim=1)  # [B, branch_output_dim*2]

        # Classification
        logits = self.classifier(fused)  # [B, num_classes]

        return logits

    def get_features(self, spec, f0):
        """
        Extract fused features without classification

        Args:
            spec: [batch, 1, freq_bins, time_frames]
            f0: [batch, 1, time_frames]
        Returns:
            features: [batch, branch_output_dim*2] - Fused features
        """
        spec_feat = self.spec_branch(spec)
        f0_feat = self.f0_branch(f0)
        fused = torch.cat([spec_feat, f0_feat], dim=1)
        return fused


# Factory function for easy model creation
def create_cnnlstm_2ff(
    num_classes: int,
    branch_output_dim: int = 512,
    fusion_hidden_dim: int = 512,
    dropout: float = 0.5
) -> CNNLSTM_2FF:
    """
    Create a CNN+LSTM 2-Feature Fusion model with default settings

    Args:
        num_classes: Number of output classes
        branch_output_dim: Output dimension of each feature branch
        fusion_hidden_dim: Hidden dimension of fusion layer
        dropout: Dropout rate

    Returns:
        model: CNNLSTM_2FF model instance
    """
    return CNNLSTM_2FF(
        num_classes=num_classes,
        branch_output_dim=branch_output_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    num_classes = 6  # Example: 6 noise types
    batch_size = 4

    # Create model
    model = create_cnnlstm_2ff(
        num_classes=num_classes,
        branch_output_dim=512,
        fusion_hidden_dim=512,
        dropout=0.5
    ).to(device)

    # Test inputs
    spec = torch.randn(batch_size, 1, 128, 126).to(device)  # Mel-spectrogram
    f0 = torch.randn(batch_size, 1, 126).to(device)         # F0 contour

    # Forward pass
    output = model(spec, f0)
    features = model.get_features(spec, f0)

    print("=" * 60)
    print("CNN+LSTM 2-Feature Fusion Model Test")
    print("=" * 60)
    print(f"Input shapes:")
    print(f"  Spectrogram: {spec.shape}")
    print(f"  F0:          {f0.shape}")
    print(f"\nOutput shapes:")
    print(f"  Logits:      {output.shape}")
    print(f"  Features:    {features.shape}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    # Test with different input sizes
    print("\nTesting with different time dimensions:")
    for time_frames in [100, 126, 200]:
        spec_test = torch.randn(2, 1, 128, time_frames).to(device)
        f0_test = torch.randn(2, 1, time_frames).to(device)
        output_test = model(spec_test, f0_test)
        print(f"  Time={time_frames}: Output shape = {output_test.shape}")
