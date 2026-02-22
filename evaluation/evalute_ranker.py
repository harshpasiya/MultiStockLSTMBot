import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr

# ── ROOT + import fix ─────────────────────────────────────────────────────────
# Structure:
#   <project_root>/
#     data/grouped_sequences.npy
#     dataset/ranking_dataset.py
#     evaluation/evaluate_ranker.py   ← this file
#     models/lstm_ranker.py
#     models/zodic_omega_ranker.pt
#     outputs/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.ranking_dataset import DailyRankingDataset
from models.lstm_ranker       import ZodicLSTMRanker

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = ROOT / "data"    / "grouped_sequences.npy"
MODEL_PATH  = ROOT / "models"  / "zodic_omega_ranker.pt"
SCORES_CSV  = ROOT / "outputs" / "eval_daily_scores.csv"
PREDICT_CSV = ROOT / "outputs" / "eval_future_predictions.csv"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_PCT  = 0.20
HORIZON    = 5
EVAL_SPLIT = 0.85    # evaluate on last 15% of dates (test set)


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(scores_np, rets_np):
    if len(scores_np) < 5:
        return np.nan, np.nan, np.nan

    ic, _ = spearmanr(scores_np, rets_np)
    ic    = float(ic) if not np.isnan(ic) else 0.0

    n_top   = max(1, int(len(scores_np) * TOP_K_PCT))
    top_idx = np.argsort(scores_np)[-n_top:]
    bot_idx = np.argsort(scores_np)[:n_top]
    tbpr    = float((rets_np[top_idx] > 0).mean())
    spread  = float(rets_np[top_idx].mean() - rets_np[bot_idx].mean())
    return ic, tbpr, spread


# ── SANITY CHECK ──────────────────────────────────────────────────────────────

def sanity_check(model, grouped, sample_dates, device):
    """Detects untrained/collapsed model before wasting time on full evaluation."""
    model.eval()
    variances = []
    sample = np.random.choice(sample_dates, size=min(20, len(sample_dates)), replace=False)
    with torch.no_grad():
        for date in sample:
            day_data = grouped[date]
            if len(day_data) < 5:
                continue
            seqs   = torch.tensor(
                np.array([s["seq"] for s in day_data]), dtype=torch.float32
            ).to(device)
            scores = model(seqs).cpu().numpy()
            variances.append(np.var(scores))

    mean_var = np.mean(variances)
    print(f"  Score variance across 20 sampled days : {mean_var:.6f}")
    if mean_var < 1e-6:
        print("  ⚠ WARNING: Near-zero variance → model is UNTRAINED or COLLAPSED")
        print("  ⚠ Run scripts/train_ranker.py first!")
    else:
        print("  ✓ Score variance looks healthy")
    return mean_var


# ── MAIN EVALUATION ───────────────────────────────────────────────────────────

def evaluate():
    print(f"Device  : {DEVICE}")
    print(f"Loading : {DATA_PATH}\n")

    grouped   = np.load(DATA_PATH, allow_pickle=True).item()
    all_dates = sorted(grouped.keys())
    n         = len(all_dates)
    test_dates = all_dates[int(EVAL_SPLIT * n):]

    print(f"Total dates  : {n}")
    print(f"Test dates   : {len(test_dates)} (last 15%)")

    n_features = grouped[all_dates[0]][0]["seq"].shape[-1]
    print(f"Features     : {n_features}\n")

    if not MODEL_PATH.exists():
        print(f"✗ Model not found at {MODEL_PATH}")
        print("  Run scripts/train_ranker.py first!")
        return

    model = ZodicLSTMRanker(n_features=n_features).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✓ Loaded model ← {MODEL_PATH}\n")

    print("── Sanity Check ──────────────────────────────────────")
    sanity_check(model, grouped, test_dates, DEVICE)

    print("\n── Per-Day Evaluation ────────────────────────────────")
    records, day_ics, day_tbprs, day_spreads = [], [], [], []

    with torch.no_grad():
        for date in test_dates:
            day_data = grouped[date]
            if len(day_data) < 5:
                continue
            seqs    = torch.tensor(
                np.array([s["seq"] for s in day_data]), dtype=torch.float32
            ).to(DEVICE)
            tickers = [s["ticker"] for s in day_data]
            rets_np = np.array([s["ret"] for s in day_data])
            scores  = model(seqs).cpu().numpy()

            ic, tbpr, spread = compute_metrics(scores, rets_np)
            day_ics.append(ic)
            day_tbprs.append(tbpr)
            day_spreads.append(spread)

            n_top     = max(1, int(len(scores) * TOP_K_PCT))
            threshold = np.sort(scores)[-n_top]

            for tkr, sc, rt in zip(tickers, scores, rets_np):
                records.append({
                    "Date":          str(date.date()),
                    "Ticker":        tkr,
                    "Score":         round(float(sc), 6),
                    "Actual_Return": round(float(rt), 6),
                    "Is_Top_Basket": bool(sc >= threshold),
                    "Day_IC":        round(float(ic), 4),
                    "Day_TBPR":      round(float(tbpr), 4),
                })

    mean_ic     = np.nanmean(day_ics)
    mean_tbpr   = np.nanmean(day_tbprs)
    mean_spread = np.nanmean(day_spreads)
    ic_std      = np.nanstd(day_ics)
    ic_pos_rate = np.nanmean([ic > 0 for ic in day_ics])
    ic_gt03     = np.nanmean([ic > 0.30 for ic in day_ics])

    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS  ({len(day_ics)} test days)")
    print(f"{'='*55}")
    print(f"  Mean IC (Spearman)       : {mean_ic:+.4f}   [target > 0.30]")
    print(f"  IC Std Dev               : {ic_std:.4f}")
    print(f"  IC > 0 rate              : {ic_pos_rate*100:.1f}%  (random = 50%)")
    print(f"  IC > 0.30 rate           : {ic_gt03*100:.1f}%")
    print(f"  Top Basket +ve Rate      : {mean_tbpr*100:.2f}%  [target > 75%]")
    print(f"  Top-minus-Bottom Spread  : {mean_spread:+.4f}")
    print(f"{'='*55}")

    pcts   = [10, 25, 50, 75, 90]
    ic_pct = np.nanpercentile(day_ics, pcts)
    print(f"\n  IC Percentiles:")
    for p, v in zip(pcts, ic_pct):
        print(f"    P{p:2d}: {v:+.4f}")

    print(f"\n{'='*55}")
    if mean_ic >= 0.30 and mean_tbpr >= 0.75:
        print("  ✓ TARGETS MET: IC > 0.30 AND TBPR > 75%")
    elif -0.05 < mean_ic < 0.05:
        print("  ✗ IC near zero → model is UNTRAINED")
        print("    ACTION: Run scripts/train_ranker.py with updated code")
    elif mean_ic < 0:
        print("  ✗ NEGATIVE IC → check target sign in feature_engineering.py")
    else:
        print(f"  ~ Partial: IC {mean_ic:.4f} — continue training or tune hyperparams")
    print(f"{'='*55}\n")

    os.makedirs(SCORES_CSV.parent, exist_ok=True)
    pd.DataFrame(records).to_csv(SCORES_CSV, index=False)
    print(f"✓ Daily scores saved  → {SCORES_CSV}")
    predict_latest(model, grouped, all_dates)


# ── FUTURE PREDICTIONS ────────────────────────────────────────────────────────

@torch.no_grad()
def predict_latest(model, grouped, all_dates):
    model.eval()
    for latest_date in reversed(all_dates):
        if len(grouped[latest_date]) >= 5:
            break

    print(f"\nGenerating predictions for: {latest_date.date()}"
          f"  (hold {HORIZON} trading days)")

    stocks  = grouped[latest_date]
    seqs    = torch.tensor(
        np.array([s["seq"] for s in stocks]), dtype=torch.float32
    ).to(DEVICE)
    tickers = [s["ticker"] for s in stocks]
    scores  = model(seqs).cpu().numpy()

    n_top    = max(1, int(len(scores) * TOP_K_PCT))
    rank_arr = (-scores).argsort().argsort() + 1

    pred_df = pd.DataFrame({
        "Ticker":          tickers,
        "Score":           scores,
        "Rank":            rank_arr,
        "Signal":          ["BUY" if r <= n_top else "NEUTRAL" for r in rank_arr],
        "Prediction_Date": str(latest_date.date()),
        "Hold_Until":      str((latest_date + pd.Timedelta(days=HORIZON)).date()),
    }).sort_values("Score", ascending=False).reset_index(drop=True)

    pred_df.to_csv(PREDICT_CSV, index=False)
    print(f"✓ Predictions saved   → {PREDICT_CSV}")

    print(f"\n{'='*50}")
    print(f"  TOP {n_top} BUY SIGNALS  (next {HORIZON} days)")
    print(f"{'='*50}")
    print(pred_df[pred_df["Signal"] == "BUY"][["Rank", "Ticker", "Score"]].to_string(index=False))
    print(f"{'='*50}\n")


if __name__ == "__main__":
    evaluate()
