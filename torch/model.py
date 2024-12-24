import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Base feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Phase-specific pathways
        self.opening_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.middle_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.endgame_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Material evaluation
        self.material_conv = nn.Conv2d(64, 32, kernel_size=1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            batch_first=True
        )
        
        # Output layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Base features
        x = self.conv1(x)
        
        # Phase-specific processing
        opening_feat = self.opening_conv(x)
        middle_feat = self.middle_conv(x)
        endgame_feat = self.endgame_conv(x)
        
        # Material evaluation
        material = self.material_conv(x)
        
        # Attention
        attention_in = x.flatten(2).permute(0, 2, 1)
        attention_out, _ = self.attention(attention_in, attention_in, attention_in)
        attention_out = attention_out.permute(0, 2, 1).view(x.shape)
        
        # Combine features
        x = x + opening_feat + middle_feat + endgame_feat + attention_out
        
        # Output
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x