import torch
import torch.nn as nn
import torch.nn.functional as F


class ZodicLSTMRanker(nn.Module):

    def __init__(self, n_features=5):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=64,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(0.15)

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(32)
        self.drop2 = nn.Dropout(0.15)

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = self.norm1(x)
        x = self.drop1(x)

        x, _ = self.lstm2(x)
        x = x[:, -1, :]   # last timestep

        x = self.norm2(x)
        x = self.drop2(x)

        x = F.gelu(self.fc1(x))
        score = self.fc2(x)

        return score.squeeze(-1)
