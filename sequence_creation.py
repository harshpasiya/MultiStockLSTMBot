import os
import numpy as np
import pandas as pd
from collections import defaultdict

LOOKBACK    = 25
HORIZON     = 5
INPUT_FILE  = "data/processed/all_stocks_features.csv"
OUTPUT_FILE = "data/grouped_sequences.npy"
META_FILE   = "data/sequence_meta.npy"   # stores n_features for model auto-config

def build_grouped_sequences(df):
    grouped      = defaultdict(list)
    feature_cols = [c for c in df.columns if c.endswith("_rank")]

    # Ensure Date is a proper column, not index
    df = df.reset_index(drop=True)
    if "Date" not in df.columns:
        raise ValueError("'Date' column missing from dataframe")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    for ticker, stock_df in df.groupby("Ticker"):
        stock_df = stock_df.sort_values("Date").reset_index(drop=True)
        if len(stock_df) < LOOKBACK + HORIZON + 5:
            continue

        features = stock_df[feature_cols].values.astype(np.float32)
        targets  = stock_df["target"].values
        dates    = stock_df["Date"].values

        for i in range(LOOKBACK, len(stock_df) - HORIZON):
            grouped[pd.Timestamp(dates[i])].append({
                "seq":    features[i - LOOKBACK: i],  # (LOOKBACK, n_features)
                "ret":    float(targets[i]),
                "ticker": ticker,
            })

    return grouped, feature_cols


def build_dataset():
    print("\nLoading processed multi-stock dataset...")
    df = pd.read_csv(
        INPUT_FILE,
        index_col=0,
        parse_dates=True,
    )
    # Bring Date back as column if it was used as index
    if df.index.name == "Date" or str(df.index.dtype) == "datetime64[ns]":
        df = df.reset_index()
    elif "Date" not in df.columns:
        df = df.reset_index()

    print(f"Total rows  : {len(df)}")
    print(f"Total stocks: {df['Ticker'].nunique()}")

    grouped, feature_cols = build_grouped_sequences(df)

    print(f"\nTotal trading days : {len(grouped)}")
    print(f"Features per step  : {len(feature_cols)}")
    print("Feature list:", feature_cols)

    counts = [len(v) for v in grouped.values()]
    print(f"\nStocks-per-day  Min:{np.min(counts)}  Max:{np.max(counts)}  "
          f"Mean:{np.mean(counts):.1f}")

    os.makedirs("data", exist_ok=True)
    np.save(OUTPUT_FILE, grouped, allow_pickle=True)
    np.save(META_FILE, {"n_features": len(feature_cols),
                        "feature_names": feature_cols}, allow_pickle=True)
    print(f"\n✓ Sequences → {OUTPUT_FILE}")
    print(f"✓ Meta      → {META_FILE}")

if __name__ == "__main__":
    build_dataset()
