import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION (LOCK THESE)
# ============================================================

LOOKBACK = 45

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

INPUT_FILE = "data/processed/all_stocks_features.csv"
OUTPUT_FILE = "data/sequences.npz"


# ============================================================
# SEQUENCE BUILDER (MULTI-STOCK SAFE)
# ============================================================

def create_sequences(df):

    X, y, tickers = [], [], []

    feature_cols = [col for col in df.columns if col.endswith("_rank")]

    for ticker in df["Ticker"].unique():

        stock_df = df[df["Ticker"] == ticker].copy()

        stock_df = stock_df.sort_index()

        if len(stock_df) < LOOKBACK + 20:
            continue

        features = stock_df[feature_cols].values
        targets = stock_df["target"].values

        for i in range(LOOKBACK, len(stock_df)):

            X.append(features[i - LOOKBACK:i])
            y.append(targets[i])
            tickers.append(ticker)

    return np.array(X), np.array(y), np.array(tickers)


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

    df = df.sort_index()

    print(f"Total rows: {len(df)}")
    print(f"Total stocks: {df['Ticker'].nunique()}")

    X, y, tickers = create_sequences(df)

    print(f"\nTotal sequences created: {len(X)}")

    # --------------------------------------------------------
    # GLOBAL TIME-BASED SPLIT (CRITICAL)
    # --------------------------------------------------------

    unique_dates = sorted(df.index.unique())

    train_cutoff = unique_dates[int(len(unique_dates) * TRAIN_SPLIT)]
    val_cutoff = unique_dates[int(len(unique_dates) * (TRAIN_SPLIT + VAL_SPLIT))]

    sequence_dates = df.index[LOOKBACK:]

    date_map = sequence_dates[:len(X)]

    train_mask = date_map <= train_cutoff
    val_mask = (date_map > train_cutoff) & (date_map <= val_cutoff)
    test_mask = date_map > val_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    ticker_test = tickers[test_mask]

    print("\nFinal Dataset Shapes:")
    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)
    print("X_test :", X_test.shape)

    os.makedirs("data", exist_ok=True)

    np.savez(
        OUTPUT_FILE,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        ticker_test=ticker_test
    )

    print(f"\nDataset saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()
