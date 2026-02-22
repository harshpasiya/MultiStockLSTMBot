import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.ranking_dataset import DailyRankingDataset
from models.lstm_ranker       import ZodicLSTMRanker

DATA_PATH           = ROOT / "data"    / "grouped_sequences.npy"
MODEL_PATH          = ROOT / "models"  / "zodic_omega_ranker.pt"
PREDICT_CSV         = ROOT / "outputs" / "future_predictions.csv"

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
LR                  = 3e-4
EPOCHS              = 150
EARLY_STOP_PATIENCE = 20
WEIGHT_DECAY        = 3e-4
TRAIN_FRAC          = 0.80
VAL_FRAC            = 0.10
PROB_THRESHOLD      = 0.45
HORIZON             = 5


class FocalLoss(nn.Module):
    """
    Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=2 downweights easy examples, forces model to learn hard ones.
    """
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma      = gamma

    def forward(self, logits, targets):
        bce     = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


def get_pos_weight(dataset):
    labels = []
    for i in range(len(dataset)):
        _, rets, _, _ = dataset.get_day_data(i)
        labels.extend(rets.numpy().tolist())
    lb = (np.array(labels) > 0.5).astype(float)
    pos, neg = lb.sum(), len(lb) - lb.sum()
    if pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    pw = neg / pos
    print(f"  Class balance : {pos:.0f} pos / {neg:.0f} neg  "
          f"({pos/(pos+neg)*100:.1f}% positive, pos_weight={pw:.2f})")
    return torch.tensor([pw], dtype=torch.float32)


@torch.no_grad()
def evaluate_dataset(model, dataset, criterion):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for i in range(len(dataset)):
        seqs, rets, _, _ = dataset.get_day_data(i)
        logits     = model(seqs.to(DEVICE))
        labels_bin = (rets > 0.5).float().to(DEVICE)
        total_loss += criterion(logits, labels_bin).item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(rets.numpy().tolist())
    preds  = [int(p > PROB_THRESHOLD) for p in all_probs]
    labels = [int(l > 0.5) for l in all_labels]
    if sum(labels) == 0:
        return total_loss / max(len(dataset), 1), 0.0, 0.0, 0.0
    return (total_loss / max(len(dataset), 1),
            f1_score(labels, preds, zero_division=0),
            precision_score(labels, preds, zero_division=0),
            recall_score(labels, preds, zero_division=0))


def train():
    print(f"Device : {DEVICE}")
    grouped   = np.load(DATA_PATH, allow_pickle=True).item()
    all_dates = sorted(grouped.keys())
    n         = len(all_dates)

    train_dates = all_dates[:int(TRAIN_FRAC * n)]
    val_dates   = all_dates[int(TRAIN_FRAC * n):int((TRAIN_FRAC + VAL_FRAC) * n)]
    test_dates  = all_dates[int((TRAIN_FRAC + VAL_FRAC) * n):]

    train_ds     = DailyRankingDataset(grouped, train_dates)
    val_ds       = DailyRankingDataset(grouped, val_dates)
    test_ds      = DailyRankingDataset(grouped, test_dates)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    n_features = grouped[all_dates[0]][0]["seq"].shape[-1]
    print(f"Features : {n_features}")
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} days")
    print("Computing class weights...")
    pos_weight = get_pos_weight(train_ds)

    model     = ZodicLSTMRanker(n_features=n_features).to(DEVICE)
    params    = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}  (GRU binary classifier)\n")

    criterion = FocalLoss(pos_weight.to(DEVICE), gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=5e-6
    )

    best_val_f1, best_state, no_improve = -np.inf, None, 0
    print(f"Training  (GRU + FocalLoss + threshold={PROB_THRESHOLD})\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_losses, ep_probs, ep_labels = [], [], []

        for seqs, rets in train_loader:
            seqs       = seqs.squeeze(0).to(DEVICE)
            seqs       = seqs + torch.randn_like(seqs) * 0.005
            labels_bin = (rets.squeeze(0) > 0.5).float().to(DEVICE)
            logits     = model(seqs)
            loss       = criterion(logits, labels_bin)
            optimizer.zero_grad();  loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss.item())
            ep_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            ep_labels.extend(labels_bin.cpu().numpy().tolist())

        scheduler.step()
        t_preds  = [int(p > PROB_THRESHOLD) for p in ep_probs]
        t_f1     = f1_score([int(l) for l in ep_labels], t_preds, zero_division=0)
        _, val_f1, val_p, val_r = evaluate_dataset(model, val_ds, criterion)

        print(f"Epoch {epoch:3d}/{EPOCHS}  Loss {np.mean(ep_losses):.4f}  |  "
              f"Train F1 {t_f1:.4f}  |  Val F1 {val_f1:.4f}  P {val_p:.3f}  R {val_r:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1;  no_improve = 0
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            os.makedirs(MODEL_PATH.parent, exist_ok=True)
            torch.save(best_state, MODEL_PATH)
            print(f"  ✓ Best saved  (Val F1 = {best_val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stop at epoch {epoch}  Best Val F1 = {best_val_f1:.4f}")
                break

    model.load_state_dict(best_state)
    _, tf1, tp, tr = evaluate_dataset(model, test_ds, criterion)
    print(f"\n{'='*52}\nTEST:  F1={tf1:.4f}  P={tp:.4f}  R={tr:.4f}\n{'='*52}")
    os.makedirs(PREDICT_CSV.parent, exist_ok=True)
    save_predictions(model, grouped, all_dates)


@torch.no_grad()
def save_predictions(model, grouped, all_dates):
    model.eval()
    for latest_date in reversed(all_dates):
        if len(grouped[latest_date]) >= 5:
            break
    stocks  = grouped[latest_date]
    seqs    = torch.tensor(np.array([s["seq"] for s in stocks]), dtype=torch.float32)
    tickers = [s["ticker"] for s in stocks]
    probs   = torch.sigmoid(model(seqs)).numpy()
    pred_df = pd.DataFrame({"Ticker": tickers, "Probability": probs,
                             "Signal": ["BUY" if p > PROB_THRESHOLD else "NEUTRAL"
                                        for p in probs]}
                           ).sort_values("Probability", ascending=False)
    pred_df.to_csv(PREDICT_CSV, index=False)
    buy = pred_df[pred_df["Signal"] == "BUY"]
    print(f"\nBUY signals (P > {PROB_THRESHOLD}): {len(buy)} stocks")
    if len(buy):
        print(buy[["Ticker","Probability"]].to_string(index=False))
    else:
        print("  No stocks pass threshold today — stay flat.")


if __name__ == "__main__":
    os.makedirs(ROOT / "models", exist_ok=True)
    os.makedirs(ROOT / "outputs", exist_ok=True)
    train()
