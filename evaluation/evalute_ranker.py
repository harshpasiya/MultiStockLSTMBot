import numpy as np
import torch
from pathlib import Path
from models.lstm_ranker import ZodicLSTMRanker


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "grouped_sequences.npy"
MODEL_PATH = ROOT / "models" / "zodic_omega_ranker.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Spearman Rank IC
# ------------------------------------------------------------
def rank_ic(pred, target):

    pred_rank = pred.argsort().argsort().float()
    tgt_rank = target.argsort().argsort().float()

    pred_rank -= pred_rank.mean()
    tgt_rank -= tgt_rank.mean()

    return (pred_rank * tgt_rank).mean() / (
        pred_rank.std() * tgt_rank.std() + 1e-8
    )


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate():

    print("Loading data...")
    grouped = np.load(DATA_PATH, allow_pickle=True).item()

    dates = sorted(grouped.keys())
    test_dates = dates[int(len(dates)*0.85):]

    model = ZodicLSTMRanker(n_features=18).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    ic_list = []
    spread_list = []
    hit_list = []

    for d in test_dates:

        stocks = grouped[d]
        if len(stocks) < 10:
            continue

        seqs = torch.tensor(np.array([s["seq"] for s in stocks]), dtype=torch.float32).to(DEVICE)
        rets = torch.tensor(np.array([s["ret"] for s in stocks]), dtype=torch.float32).to(DEVICE)

        scores = model(seqs)

        # Rank IC
        ic = rank_ic(scores, rets).item()
        ic_list.append(ic)

        # Top vs Bottom spread
        k = 3
        top_idx = torch.topk(scores, k).indices
        bot_idx = torch.topk(scores, k, largest=False).indices

        top_ret = rets[top_idx].mean().item()
        bot_ret = rets[bot_idx].mean().item()

        spread = top_ret - bot_ret
        spread_list.append(spread)

        # Hit rate (top basket positive)
        hit_list.append(1 if top_ret > 0 else 0)

    print("\n===== RESULTS =====")
    print(f"Mean Rank IC: {np.mean(ic_list):.4f}")
    print(f"Top-3 minus Bottom-3 Spread: {np.mean(spread_list):.4f}")
    print(f"Top Basket Positive Rate: {np.mean(hit_list)*100:.2f}%")
    print("===================")


if __name__ == "__main__":
    evaluate()