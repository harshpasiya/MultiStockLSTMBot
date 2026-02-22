import torch
import torch.nn as nn
import torch.nn.functional as F


class ZodicLSTMRanker(nn.Module):
    """
    GRU-based binary classifier.
    Processes the full 25-day sequence — NOT just the last timestep.

    Output  : raw logit  (apply sigmoid during inference)
    Loss    : FocalLoss  (defined in train_ranker.py)
    Entry   : prob = sigmoid(model(x)) > 0.45
    """

    def __init__(self, n_features: int, hidden: int = 48):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.gru  = nn.GRU(n_features, hidden, num_layers=1,
                            batch_first=True, dropout=0.0)
        self.drop = nn.Dropout(0.30)
        self.fc1  = nn.Linear(hidden, 24)
        self.fc2  = nn.Linear(24, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)           # (batch, 25, n_features)
        _, h_n = self.gru(x)             # h_n: (1, batch, hidden)
        h = h_n.squeeze(0)               # (batch, hidden)
        h = self.drop(F.gelu(self.fc1(h)))
        return self.fc2(h).squeeze(-1)   # raw logit (batch,)
