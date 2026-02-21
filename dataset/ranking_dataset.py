import numpy as np
import torch
from torch.utils.data import Dataset


class DailyRankingDataset(Dataset):

    def __init__(self, grouped_data, dates):

        self.grouped = grouped_data
        self.dates = [d for d in dates if len(grouped_data[d]) >= 5]

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):

        date = self.dates[idx]
        stocks = self.grouped[date]

        seqs = [s["seq"] for s in stocks]
        rets = [s["ret"] for s in stocks]

        seqs = torch.tensor(np.array(seqs), dtype=torch.float32)
        rets = torch.tensor(np.array(rets), dtype=torch.float32)

        return seqs, rets