import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION (LOCK THESE)
# ============================================================

LOOKBACK = 30
HORIZON = 5

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

PROCESSED_DIR = "data/processed"
OUTPUT_FILE = "data/sequences.npz"


# ============================================================
# SEQUENCE BUILDER
# ============================================================

def create_sequences(features_df):
    """
    Create sequences and forward returns for one stock.
    """

    X, y = [], []

    close_prices = features_df["ret_1"].copy()  # using return-based target proxy

    # Reconstruct actual forward return target using cumulative returns
    # Forward log return approximation
    forward_target = (
        features_df["ret_1"]
        .rolling(HORIZON)
        .sum()
        .shift(-HORIZON)
    )

    features_only = features_df.values

    for i in range(LOOKBACK, len(features_df) - HORIZON):
        X.append(features_only[i - LOOKBACK:i])
        y.append(forward_target.iloc[i])

    return np.array(X), np.array(y)


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_stacked_dataset():

    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []

    processed_files = [
        f for f in os.listdir(PROCESSED_DIR)
        if f.endswith("_features.csv")
    ]

    print(f"\nFound {len(processed_files)} processed stocks\n")

    for file in processed_files:

        symbol = file.replace("_features.csv", "")
        print(f"Building sequences for {symbol}...", end=" ")

        df = pd.read_csv(
            os.path.join(PROCESSED_DIR, file),
            index_col=0,
            parse_dates=True
        )

        if len(df) < LOOKBACK + HORIZON + 50:
            print("Skipped (too short)")
            continue

        X, y = create_sequences(df)

        # Remove NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        n = len(X)

        train_end = int(n * TRAIN_SPLIT)
        val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        all_X_train.append(X_train)
        all_y_train.append(y_train)

        all_X_val.append(X_val)
        all_y_val.append(y_val)

        all_X_test.append(X_test)
        all_y_test.append(y_test)

        print(f"OK ({n} sequences)")

    # Stack all stocks together
    X_train = np.vstack(all_X_train)
    y_train = np.concatenate(all_y_train)

    X_val = np.vstack(all_X_val)
    y_val = np.concatenate(all_y_val)

    X_test = np.vstack(all_X_test)
    y_test = np.concatenate(all_y_test)

    print("\nFinal Dataset Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    os.makedirs("data", exist_ok=True)

    np.savez(
        OUTPUT_FILE,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )

    print(f"\nDataset saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    build_stacked_dataset()
