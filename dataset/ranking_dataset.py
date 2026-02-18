import numpy as np
import torch
from torch.utils.data import Dataset
import random


class PairwiseRankingDataset(Dataset):

    def __init__(self, grouped_data, dates, samples_per_epoch=50000):

        self.grouped_data = grouped_data
        self.dates = dates
        self.samples_per_epoch = samples_per_epoch

        self.valid_dates = [
            d for d in dates if len(grouped_data[d]) >= 2
        ]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):

        # Randomly choose a date
        date = random.choice(self.valid_dates)

        stocks = self.grouped_data[date]

        # Randomly choose two different stocks
        i, j = random.sample(range(len(stocks)), 2)

        A = stocks[i]
        B = stocks[j]

        seq_A = torch.tensor(A["seq"], dtype=torch.float32)
        seq_B = torch.tensor(B["seq"], dtype=torch.float32)

        label = 1.0 if A["ret"] > B["ret"] else 0.0
        label = torch.tensor(label, dtype=torch.float32)

        return seq_A, seq_B, label
