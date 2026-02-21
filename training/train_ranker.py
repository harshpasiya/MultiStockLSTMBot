import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset.ranking_dataset import DailyRankingDataset
from models.lstm_ranker import ZodicLSTMRanker


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "grouped_sequences.npy"
MODEL_PATH = ROOT / "models" / "zodic_omega_ranker.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
LR = 1e-3


# ============================================================
# CORRELATION LOSS (LISTWISE RANK OPTIMIZATION)
# ============================================================

def corr_loss(pred, target):
    """
    Negative Pearson correlation
    Approximates RankIC optimization
    """

    pred = pred - pred.mean()
    target = target - target.mean()

    cov = (pred * target).mean()
    pred_std = pred.std() + 1e-8
    target_std = target.std() + 1e-8

    corr = cov / (pred_std * target_std)

    return -corr


# ============================================================
# EVALUATION METRIC
# ============================================================

@torch.no_grad()
def evaluate(model, dataset):

    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ic_list = []

    for seqs, rets in loader:

        seqs = seqs.squeeze(0).to(DEVICE)
        rets = rets.squeeze(0).to(DEVICE)

        scores = model(seqs)

        ic = -corr_loss(scores, rets).item()
        ic_list.append(ic)

    return np.mean(ic_list)


# ============================================================
# TRAINING LOOP
# ============================================================

def train():

    print("\nLoading grouped dataset...")
    grouped = np.load(DATA_PATH, allow_pickle=True).item()

    all_dates = sorted(grouped.keys())
    n = len(all_dates)

    train_dates = all_dates[:int(0.7*n)]
    val_dates = all_dates[int(0.7*n):int(0.85*n)]

    train_ds = DailyRankingDataset(grouped, train_dates)
    val_ds = DailyRankingDataset(grouped, val_dates)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    model = ZodicLSTMRanker(n_features=18).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_ic = -1

    print("\nTraining (IC optimization)...\n")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for seqs, rets in train_loader:

            seqs = seqs.squeeze(0).to(DEVICE)
            rets = rets.squeeze(0).to(DEVICE)

            scores = model(seqs)

            loss = corr_loss(scores, rets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        val_ic = evaluate(model, val_ds)

        print(f"Epoch {epoch+1:02d} | TrainLoss {total_loss/len(train_loader):.4f} | ValIC {val_ic:.4f}")

        if val_ic > best_ic:
            best_ic = val_ic
            torch.save(model.state_dict(), MODEL_PATH)
            print("✔ Saved Best Model")

    print("\nBest Validation IC:", best_ic)


if __name__ == "__main__":
    os.makedirs(ROOT / "models", exist_ok=True)
    train()