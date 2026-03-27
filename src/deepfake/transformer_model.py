import torch
import torch.nn as nn


class DeepfakeTransformer(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=2, seq_len=20):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Add positional encoding
        x = x + self.pos_embedding

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.classifier(x)