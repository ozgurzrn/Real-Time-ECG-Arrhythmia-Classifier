import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        weights = self.attention(x) # (batch, seq_len, 1)
        # Context vector: Weighted sum of LSTM outputs
        context = torch.sum(x * weights, dim=1) # (batch, hidden_dim)
        return context, weights

class SOTAEnsembleModel(nn.Module):
    """
    SOTA Ensemble Model for ECG Arrhythmia Classification.
    Combines:
    1. ResNet-based 1D-CNN for feature extraction.
    2. Bidirectional LSTM for temporal dependency capture.
    3. Attention Mechanism for focusing on critical segments (e.g., QRS complex).
    """
    def __init__(self, num_classes=5):
        super(SOTAEnsembleModel, self).__init__()
        
        # --- CNN Feature Extractor (ResNet Style) ---
        self.in_channels = 32
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        
        # Flattening size calculation:
        # Input: (B, 1, 187)
        # Conv1: (B, 32, 187)
        # MaxPool: (B, 32, 94)
        # Layer1: (B, 32, 94)
        # Layer2: (B, 64, 47)
        # CNN Output: (B, 64, 47) -> Permute to (B, 47, 64) for LSTM
        
        cnn_out_channels = 64
        
        # --- RNN Layer ---
        self.hidden_dim = 64
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels, 
            hidden_size=self.hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        # Bidirectional -> 2 * hidden_dim
        self.lstm_out_dim = self.hidden_dim * 2
        
        # --- Attention Layer ---
        self.attention = AttentionBlock(self.lstm_out_dim)
        
        # --- Classifier ---
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.lstm_out_dim, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, seq_len) or (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # 1. CNN Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x) # Output: (B, 64, L)
        
        # 2. Reshape for LSTM: (B, L, Features)
        x = x.permute(0, 2, 1) # (B, L, 64)
        
        # 3. LSTM
        x, _ = self.lstm(x) # Output: (B, L, 2*hidden_dim)
        
        # 4. Attention
        context, weights = self.attention(x) # Output: (B, 2*hidden_dim)
        
        # 5. Classification
        x = self.dropout(context)
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    # Test model
    model = SOTAEnsembleModel()
    x = torch.randn(16, 187)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
