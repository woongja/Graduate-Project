"""
CNN8RNN 3-Feature Fusion (3FF) Base Model
Fuses Spectrogram + MFCC + F0 features using CNN8RNN architecture with simple concatenation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def _init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    """CNN block used in CNN8RNN architecture"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3), (1,1), (1,1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3,3), (1,1), (1,1), bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(self, x, pool_size=(2,2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg+max':
            x = F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, pool_size)
        else:
            x = F.max_pool2d(x, pool_size)
        return x


class SpectrogramBranch(nn.Module):
    """CNN8 + GRU branch for Spectrogram feature extraction"""
    def __init__(self, output_dim=512, n_mels=128):
        super().__init__()

        # CNN8 backbone
        self.bn0 = nn.BatchNorm2d(n_mels)  # Match input mel bins
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.rnn = nn.GRU(512, 256, bidirectional=True, batch_first=True)

        # Output projection
        self.fc_out = nn.Linear(512, output_dim)

        _init_bn(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_out)

    def forward(self, x):
        """
        Args:
            x: [B, 1, freq_bins, time_frames] - Mel Spectrogram (e.g., [B, 1, 128, T])
        Returns:
            features: [B, output_dim]
        """
        # Transpose for BatchNorm: [B, 1, F, T] -> [B, F, T, 1]
        x = x.squeeze(1)       # [B, F, T]
        x = x.unsqueeze(-1)    # [B, F, T, 1]
        x = self.bn0(x)        # BatchNorm on F dimension
        x = x.squeeze(-1)      # [B, F, T]
        x = x.unsqueeze(1)     # [B, 1, F, T]

        # CNN8 blocks
        x = self.conv_block1(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)

        # Temporal feature extraction
        x = torch.mean(x, dim=3)  # [B, 512, T']
        x = x.transpose(1, 2)     # [B, T', 512]
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))  # [B, T', 512]
        x, _ = self.rnn(x)        # [B, T', 512]

        # Temporal pooling
        x = x.mean(dim=1)         # [B, 512]
        x = self.fc_out(x)        # [B, output_dim]

        return x


class MFCCBranch(nn.Module):
    """CNN8 + GRU branch for MFCC feature extraction"""
    def __init__(self, output_dim=512, n_mfcc=40):
        super().__init__()

        # Smaller CNN for MFCC (lower dimension)
        self.bn0 = nn.BatchNorm2d(n_mfcc)  # Match input MFCC dims
        self.conv_block1 = ConvBlock(1, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.conv_block4 = ConvBlock(128, 256)

        self.fc1 = nn.Linear(256, 256, bias=True)
        self.rnn = nn.GRU(256, 128, bidirectional=True, batch_first=True)

        # Output projection
        self.fc_out = nn.Linear(256, output_dim)

        _init_bn(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_out)

    def forward(self, x):
        """
        Args:
            x: [B, 1, mfcc_dims, time_frames] - MFCC (e.g., [B, 1, 40, T])
        Returns:
            features: [B, output_dim]
        """
        # Transpose for BatchNorm: [B, 1, M, T] -> [B, M, T, 1]
        x = x.squeeze(1)       # [B, M, T]
        x = x.unsqueeze(-1)    # [B, M, T, 1]
        x = self.bn0(x)        # BatchNorm on M dimension
        x = x.squeeze(-1)      # [B, M, T]
        x = x.unsqueeze(1)     # [B, 1, M, T]

        x = self.conv_block1(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1,2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x, _ = self.rnn(x)

        x = x.mean(dim=1)
        x = self.fc_out(x)

        return x


class F0Branch(nn.Module):
    """1D CNN + GRU branch for F0 feature extraction"""
    def __init__(self, output_dim=512):
        super().__init__()

        # 1D CNN for F0 contour
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        self.rnn = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        self.fc_out = nn.Linear(128, output_dim)

        _init_layer(self.fc_out)

    def forward(self, x):
        """
        Args:
            x: [B, 1, time_frames] - F0 contour
        Returns:
            features: [B, output_dim]
        """
        x = self.conv(x)              # [B, 128, T']
        x = x.transpose(1, 2)         # [B, T', 128]
        x, _ = self.rnn(x)            # [B, T', 128]
        x = x.mean(dim=1)             # [B, 128]
        x = self.fc_out(x)            # [B, output_dim]

        return x


class CNN8RNN_3FF_Base(nn.Module):
    """
    CNN8RNN-based 3-Feature Fusion Network (Base version with simple concatenation)

    Architecture:
    - Spectrogram Branch: CNN8 + BiGRU
    - MFCC Branch: CNN8 (smaller) + BiGRU
    - F0 Branch: 1D CNN + BiGRU
    - Fusion: Simple concatenation + MLP classifier
    """

    def __init__(
        self,
        num_classes: int,
        branch_output_dim: int = 512,
        dropout: float = 0.5
    ):
        super().__init__()

        # Feature extraction branches
        self.spec_branch = SpectrogramBranch(output_dim=branch_output_dim)
        self.mfcc_branch = MFCCBranch(output_dim=branch_output_dim)
        self.f0_branch = F0Branch(output_dim=branch_output_dim)

        # Fusion dimensions
        fused_dim = branch_output_dim * 3  # 1536

        # Classifier: 1536 → 1024 → 512 → num_classes
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
        """
        Args:
            spec: [batch, 1, freq_bins, time_frames] - Mel Spectrogram
            mfcc: [batch, 1, mfcc_dims, time_frames] - MFCC
            f0: [batch, 1, time_frames] - F0 contour
        Returns:
            logits: [batch, num_classes]
        """
        # Extract features from each branch
        spec_feat = self.spec_branch(spec)  # [B, branch_output_dim]
        mfcc_feat = self.mfcc_branch(mfcc)  # [B, branch_output_dim]
        f0_feat = self.f0_branch(f0)        # [B, branch_output_dim]

        # Simple concatenation
        fused = torch.cat([spec_feat, mfcc_feat, f0_feat], dim=1)  # [B, branch_output_dim*3]

        # Classification
        logits = self.classifier(fused)  # [B, num_classes]

        return logits


def create_cnn8rnn_3ff_base(
    num_classes: int,
    branch_output_dim: int = 512,
    dropout: float = 0.5
):
    return CNN8RNN_3FF_Base(
        num_classes=num_classes,
        branch_output_dim=branch_output_dim,
        dropout=dropout
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_cnn8rnn_3ff_base(num_classes=10).to(device)

    # Test inputs
    spec = torch.randn(4, 1, 64, 126).to(device)   # Mel-spectrogram (64 mel bins)
    mfcc = torch.randn(4, 1, 40, 126).to(device)   # MFCC (40 coefficients)
    f0 = torch.randn(4, 1, 126).to(device)         # F0 contour

    output = model(spec, mfcc, f0)

    print("=" * 60)
    print("CNN8RNN 3FF Base Model Test")
    print("=" * 60)
    print(f"Input shapes:")
    print(f"  Spectrogram: {spec.shape}")
    print(f"  MFCC:        {mfcc.shape}")
    print(f"  F0:          {f0.shape}")
    print(f"\nOutput shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
