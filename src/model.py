import torch
import torch.nn as nn
from config import D_MODEL, N_HEADS, N_LAYERS, N_CLASSES

class ModerationLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(D_MODEL, N_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
