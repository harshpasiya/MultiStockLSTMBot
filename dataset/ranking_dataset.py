import numpy as np
import torch
from torch.utils.data import Dataset

class DailyRankingDataset(Dataset):

    def __init__(self, grouped_data, dates, min_stocks=5):
        self.grouped = grouped_data
        self.dates   = [d for d in dates if len(grouped_data[d]) >= min_stocks]

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        seqs, rets, _, _ = self.get_day_data(idx)   # ← calls get_day_data
        return seqs, rets

    def get_day_data(self, idx):                     # ← THIS must exist
        date    = self.dates[idx]
        stocks  = self.grouped[date]
        seqs    = torch.tensor(np.array([s["seq"] for s in stocks]), dtype=torch.float32)
        rets    = torch.tensor(np.array([s["ret"] for s in stocks]), dtype=torch.float32)
        tickers = [s["ticker"] for s in stocks]
        return seqs, rets, tickers, date
