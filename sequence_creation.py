import numpy as np
import pandas as pd
from collections import defaultdict

LOOKBACK = 45
HORIZON = 5

INPUT_FILE = "data/processed/all_stocks_features.csv"
OUTPUT_FILE = "data/grouped_sequences.npy"


def build_grouped_sequences(df):

    grouped = defaultdict(list)

    feature_cols = [c for c in df.columns if c.endswith("_rank")]

    df = df.sort_values(["Ticker", "Date"])

    for ticker, sdf in df.groupby("Ticker"):

        sdf = sdf.reset_index(drop=True)

        if len(sdf) < LOOKBACK + HORIZON + 5:
            continue

        features = sdf[feature_cols].values
        targets = sdf["target"].values
        dates = sdf["Date"].values

        for i in range(LOOKBACK, len(sdf) - HORIZON):

            seq = features[i-LOOKBACK:i].astype(np.float32)
            ret = float(targets[i])
            date = pd.Timestamp(dates[i])



            grouped[date].append({
                "seq": seq,
                "ret": ret,
                "ticker": ticker
            })

    # remove weak cross-section days
    cleaned = {d:v for d,v in grouped.items() if len(v) >= 10}

    return cleaned


def build_dataset():

    print("Loading processed dataset...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    grouped = build_grouped_sequences(df)

    print("Total trading days:", len(grouped))

    counts = [len(v) for v in grouped.values()]
    print("Min stocks/day:", min(counts))
    print("Max stocks/day:", max(counts))
    print("Mean stocks/day:", sum(counts)/len(counts))

    np.save(OUTPUT_FILE, grouped, allow_pickle=True)

    print("Saved →", OUTPUT_FILE)


if __name__ == "__main__":
    build_dataset()