import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG (LOCK THESE UNLESS RETRAINING MODEL)
# ============================================================

RET_WINDOWS = [1, 3, 5]
MA_WINDOWS = [10, 20]
MFI_PERIOD = 14
ATR_PERIOD = 14
VOL_WINDOW = 20
VOL_CHANGE_WINDOW = 5

FEATURE_ORDER = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ma_dist_10",
    "ma_dist_20",
    "mfi_14",
    "atr_14",
    "rolling_vol_20",
    "vol_change_5"
]


# ============================================================
# INDICATORS
# ============================================================

def calculate_mfi(df, period=14):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]

    positive_flow = []
    negative_flow = []

    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i - 1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])

    positive_flow = pd.Series(positive_flow, index=df.index[1:])
    negative_flow = pd.Series(negative_flow, index=df.index[1:])

    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()

    money_ratio = positive_mf / negative_mf.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))

    mfi = mfi.reindex(df.index)

    return mfi


def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    atr = true_range.rolling(period).mean()
    return atr


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    df = df.copy()

    # -------------------------
    # Returns
    # -------------------------
    for w in RET_WINDOWS:
        df[f"ret_{w}"] = np.log(df["Close"] / df["Close"].shift(w))

    # -------------------------
    # Moving average distance
    # -------------------------
    for w in MA_WINDOWS:
        ma = df["Close"].rolling(w).mean()
        df[f"ma_dist_{w}"] = (df["Close"] - ma) / ma

    # -------------------------
    # MFI (normalized to [-1, 1])
    # -------------------------
    df["mfi_14"] = calculate_mfi(df, MFI_PERIOD)
    df["mfi_14"] = (df["mfi_14"] - 50) / 50

    # -------------------------
    # ATR (normalized)
    # -------------------------
    df["atr_14"] = calculate_atr(df, ATR_PERIOD)
    df["atr_14"] = df["atr_14"] / df["Close"]

    # -------------------------
    # Rolling volatility
    # -------------------------
    df["rolling_vol_20"] = (
        np.log(df["Close"] / df["Close"].shift(1))
        .rolling(VOL_WINDOW)
        .std()
    )

    # -------------------------
    # Volume change
    # -------------------------
    df["vol_change_5"] = (
        df["Volume"] /
        df["Volume"].rolling(VOL_CHANGE_WINDOW).mean()
    )

    # -------------------------
    # Clean
    # -------------------------
    df = df.dropna()

    # Enforce feature order
    df = df[FEATURE_ORDER]

    return df


# ============================================================
# MAIN LOOP (MULTI-STOCK SAFE)
# ============================================================

def process_all_raw_data():

    os.makedirs("data/processed", exist_ok=True)

    raw_files = [f for f in os.listdir("data") if f.endswith("_raw.csv")]

    print(f"\nProcessing {len(raw_files)} stocks...\n")

    for file in raw_files:
        try:
            symbol = file.replace("_raw.csv", "")
            print(f"Processing {symbol}...", end=" ", flush=True)

            df = pd.read_csv(
                f"data/{file}",
                index_col=0
            )

            # Clean date column explicitly
            df.index = pd.to_datetime(
                df.index,
                format="%Y-%m-%d",
                errors="coerce"
            )

            # Force numeric columns
            numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                    )
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop completely empty rows
            df = df.dropna(subset=["Close", "High", "Low", "Volume"])

            features = engineer_features(df)

            if len(features) < 300:
                print("Skipped (too few rows)")
                continue

            features.to_csv(f"data/processed/{symbol}_features.csv")

            print(f"OK ({len(features)} rows)")

        except Exception as e:
            print(f"Error: {e}")

    print("\nFeature engineering completed.\n")


if __name__ == "__main__":
    process_all_raw_data()
