"""
CNN+LSTM 3-Feature Fusion (3FF) Model with Cross-Modal Attention
Fuses Spectrogram + MFCC + F0 features using cross-modal attention mechanism
"""
import torch
import torch.nn as nn
import math


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


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention mechanism
    Allows features to interact and attend to each other
    Based on multi-head self-attention from Transformer
    """
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, spec_feat, mfcc_feat, f0_feat):
        """
        Args:
            spec_feat: [batch, feature_dim]
            mfcc_feat: [batch, feature_dim]
            f0_feat: [batch, feature_dim]
        Returns:
            attended_features: [batch, feature_dim*3] - Cross-attended features
            attention_weights: [batch, num_heads, 3, 3] - Attention maps
        """
        batch_size = spec_feat.size(0)

        # Stack features: [B, 3, feature_dim]
        x = torch.stack([spec_feat, mfcc_feat, f0_feat], dim=1)
        residual = x

        # Linear projections and reshape for multi-head attention
        # [B, 3, feature_dim] -> [B, 3, num_heads, head_dim] -> [B, num_heads, 3, head_dim]
        Q = self.query(x).view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # [B, num_heads, 3, head_dim] @ [B, num_heads, head_dim, 3] -> [B, num_heads, 3, 3]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [B, num_heads, 3, 3] @ [B, num_heads, 3, head_dim] -> [B, num_heads, 3, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        # [B, num_heads, 3, head_dim] -> [B, 3, num_heads, head_dim] -> [B, 3, feature_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 3, self.feature_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection + Layer norm
        output = self.layer_norm(residual + self.dropout(attn_output))

        # Flatten to [B, feature_dim*3]
        output_flat = output.reshape(batch_size, -1)

        return output_flat, attn_weights


class CNNLSTM_3FF_CrossModal(nn.Module):
    """
    3-Feature Fusion Network with Cross-Modal Attention

    Key improvement: Uses multi-head cross-modal attention to allow
    features to interact and complement each other
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
        num_attention_heads: int = 8,
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

        # Cross-modal attention mechanism
        self.cross_modal_attention = CrossModalAttention(
            feature_dim=branch_output_dim,
            num_heads=num_attention_heads,
            dropout=0.1
        )

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

        # Apply cross-modal attention
        fused, self.attention_weights = self.cross_modal_attention(spec_feat, mfcc_feat, f0_feat)

        # Classification
        logits = self.classifier(fused)

        return logits

    def get_attention_weights(self):
        """Get the last computed attention weights"""
        return self.attention_weights if hasattr(self, 'attention_weights') else None


def create_cnnlstm_3ff_crossmodal(
    num_classes: int,
    branch_output_dim: int = 512,
    num_attention_heads: int = 8,
    dropout: float = 0.5
):
    return CNNLSTM_3FF_CrossModal(
        num_classes=num_classes,
        branch_output_dim=branch_output_dim,
        num_attention_heads=num_attention_heads,
        dropout=dropout
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_cnnlstm_3ff_crossmodal(num_classes=10).to(device)

    spec = torch.randn(4, 1, 128, 126).to(device)
    mfcc = torch.randn(4, 1, 40, 126).to(device)
    f0 = torch.randn(4, 1, 126).to(device)

    output = model(spec, mfcc, f0)
    attn = model.get_attention_weights()

    print("=" * 60)
    print("CNN+LSTM 3FF with Cross-Modal Attention")
    print("=" * 60)
    print(f"Output: {output.shape}")
    print(f"Attention weights: {attn.shape}")
    print(f"Attention map (head 0, batch 0):")
    print(attn[0, 0])  # [3, 3] matrix showing feature interactions
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
