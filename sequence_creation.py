import os
import numpy as np
import pandas as pd
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

LOOKBACK = 45
HORIZON = 5

INPUT_FILE = "data/processed/all_stocks_features.csv"
OUTPUT_FILE = "data/grouped_sequences.npy"


# ============================================================
# GROUPED SEQUENCE BUILDER (CRITICAL FOR RANKING)
# ============================================================

def build_grouped_sequences(df):

    grouped = defaultdict(list)

    feature_cols = [c for c in df.columns if c.endswith("_rank")]

    # ensure correct ordering
    df = df.sort_values(["Ticker", df.index.name])

    for ticker, stock_df in df.groupby("Ticker"):

        stock_df = stock_df.sort_index().reset_index()

        if len(stock_df) < LOOKBACK + HORIZON + 5:
            continue

        features = stock_df[feature_cols].values
        targets = stock_df["target"].values
        dates = stock_df.iloc[:,0].values   # date column after reset_index

        for i in range(LOOKBACK, len(stock_df) - HORIZON):

            seq = features[i - LOOKBACK:i].astype(np.float32)
            fwd_ret = float(targets[i])
            date = pd.Timestamp(dates[i])

            grouped[date].append({
                "seq": seq,
                "ret": fwd_ret,
                "ticker": ticker
            })

    return grouped


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_dataset():

    print("\nLoading processed multi-stock dataset...\n")

    df = pd.read_csv(
        INPUT_FILE,
        index_col=0,
        parse_dates=True
    )

    print(f"Total rows: {len(df)}")
    print(f"Total stocks: {df['Ticker'].nunique()}")

    grouped = build_grouped_sequences(df)

    print(f"\nTotal trading days created: {len(grouped)}")

    # Diagnostics (VERY IMPORTANT)
    counts = [len(v) for v in grouped.values()]
    print("\nStocks per day stats:")
    print("Min :", np.min(counts))
    print("Max :", np.max(counts))
    print("Mean:", np.mean(counts))

    os.makedirs("data", exist_ok=True)

    np.save(OUTPUT_FILE, grouped, allow_pickle=True)

    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()
