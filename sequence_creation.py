import os
import numpy as np
import pandas as pd
from collections import defaultdict

LOOKBACK    = 45        # encoder sees 45 days of history
HORIZON     = 5         # decoder predicts 5-day path
INPUT_FILE  = "data/processed/all_stocks_features.csv"
OUTPUT_FILE = "data/grouped_sequences.npy"

FEATURE_COLS = [
    "ret_5", "ret_20", "momentum_skip",       # ① Momentum
    "rsi_14", "bb_pct", "dist_ma50",          # ② Mean Reversion
    "adx_14", "macd_hist", "trend_slope",     # ③ Trend Quality
    "atr_pct", "vol_ratio",                   # ④ Volatility
    "vol_spike", "obv_slope",                 # ⑤ Volume / Flow
    "dist_52w_high", "candle_strength",       # ⑥ Structure
    "rel_strength",                           # ⑥ Alpha
]
TARGET_COLS = [f"target_d{n}" for n in range(1, HORIZON + 1)]


def build_grouped_sequences(df: pd.DataFrame) -> dict:
    """
    Returns grouped[date] = list of dicts:
      {
        "seq"    : (45, 16) float32  —  encoder input (45 days of features)
        "path"   : (5,)    float32  —  decoder target (cumulative 5-day return path)
        "ticker" : str
      }

    Each sample is indexed by the ENTRY DATE (last encoder timestep).
    On that date, the backtest will query the model for predictions.

    path[0] = (close_t+1 / close_t) - 1   ← 1-day cumulative return
    path[1] = (close_t+2 / close_t) - 1   ← 2-day cumulative return
    path[2] = (close_t+3 / close_t) - 1   ← 3-day cumulative return
    path[3] = (close_t+4 / close_t) - 1   ← 4-day cumulative return
    path[4] = (close_t+5 / close_t) - 1   ← 5-day cumulative return
    """
    grouped = defaultdict(list)

    for ticker, stock_df in df.groupby("Ticker"):
        stock_df = stock_df.sort_values("Date").reset_index(drop=True)

        # Need at least LOOKBACK + HORIZON + buffer rows
        if len(stock_df) < LOOKBACK + HORIZON + 5:
            continue

        features = stock_df[FEATURE_COLS].values.astype(np.float32)  # (T, 16)
        targets  = stock_df[TARGET_COLS].values.astype(np.float32)   # (T, 5)
        dates    = stock_df["Date"].values

        for i in range(LOOKBACK, len(stock_df) - HORIZON):
            seq  = features[i - LOOKBACK : i]   # (45, 16)
            path = targets[i]                   # (5,)
            date = pd.Timestamp(dates[i])

            # Skip rows with any NaN (z-score warmup period)
            if np.isnan(seq).any() or np.isnan(path).any():
                continue

            grouped[date].append({
                "seq":    seq,
                "path":   path,
                "ticker": ticker,
            })

    return dict(grouped)


def build_dataset():
    print(f"Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    print(f"  Rows        : {len(df):,}")
    print(f"  Stocks      : {df['Ticker'].nunique()}")
    print(f"  Features    : {len(FEATURE_COLS)}")
    print(f"  Targets     : {len(TARGET_COLS)}  (path d1..d5)")
    print(f"  LOOKBACK    : {LOOKBACK} days")
    print()

    grouped   = build_grouped_sequences(df)
    all_dates = sorted(grouped.keys())
    counts    = [len(grouped[d]) for d in all_dates]

    print(f"Grouped sequences:")
    print(f"  Trading days: {len(all_dates)}")
    print(f"  Date range  : {all_dates[0].date()} to {all_dates[-1].date()}")
    print(f"  Stocks/day  : min={min(counts)}  max={max(counts)}  "
          f"mean={np.mean(counts):.0f}")
    print()

    # Sanity check
    mid     = all_dates[len(all_dates) // 2]
    sample  = grouped[mid][0]
    print(f"Sanity check (date={mid.date()}):")
    print(f"  seq.shape   : {sample['seq'].shape}   "
          f"expected ({LOOKBACK}, {len(FEATURE_COLS)})")
    print(f"  path.shape  : {sample['path'].shape}   expected ({HORIZON},)")
    print(f"  path values : {np.round(sample['path'] * 100, 2)}%")
    print(f"  ticker      : {sample['ticker']}")
    print()

    # Train / Val / Test split preview
    n       = len(all_dates)
    n_train = int(0.80 * n)
    n_val   = int(0.10 * n)
    n_test  = n - n_train - n_val

    print(f"Split preview (80 / 10 / 10):")
    print(f"  Train : {n_train} days  "
          f"({all_dates[0].date()} to {all_dates[n_train-1].date()})")
    print(f"  Val   : {n_val} days  "
          f"({all_dates[n_train].date()} to "
          f"{all_dates[n_train + n_val - 1].date()})")
    print(f"  Test  : {n_test} days  "
          f"({all_dates[n_train + n_val].date()} to {all_dates[-1].date()})")
    print()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.save(OUTPUT_FILE, grouped, allow_pickle=True)

    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"✓  Saved  -> {OUTPUT_FILE}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build_dataset()
