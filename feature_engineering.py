import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

LOOKBACK_Z = 100
VOL_WINDOW = 20
MFI_PERIOD = 14
HORIZON = 5

FEATURE_ORDER = [
    "ret_5_rank",
    "ma_dist_20_rank",
    "mfi_14_rank",
    "rolling_vol_20_rank",
    "vol_regime_rank"
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

    money_ratio = positive_mf / (negative_mf + 1e-8)
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi.reindex(df.index)


def rolling_zscore(series, window=100):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-8)


# ============================================================
# FEATURE ENGINEERING (PER STOCK)
# ============================================================

def engineer_features(df, ticker):

    df = df.copy()

    # Returns
    df["ret_1"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ret_5"] = np.log(df["Close"] / df["Close"].shift(5))

    # MA Distance
    ma20 = df["Close"].rolling(20).mean()
    df["ma_dist_20"] = (df["Close"] - ma20) / (ma20 + 1e-8)

    # MFI
    df["mfi_14"] = calculate_mfi(df, MFI_PERIOD)
    df["mfi_14"] = (df["mfi_14"] - 50) / 50

    # Volatility
    df["rolling_vol_20"] = df["ret_1"].rolling(VOL_WINDOW).std()
    df["rolling_vol_50"] = df["ret_1"].rolling(50).std()
    df["rolling_vol_200"] = df["ret_1"].rolling(200).std()

    df["vol_regime"] = df["rolling_vol_50"] / (df["rolling_vol_200"] + 1e-8)
    df.drop(columns=["rolling_vol_50", "rolling_vol_200"], inplace=True)

    # Forward return (NO cross-sectional logic here)
    df["fwd_return"] = np.log(
        df["Close"].shift(-HORIZON) / df["Close"]
    )

    # Per-stock rolling normalization
    feature_cols = [
        "ret_5",
        "ma_dist_20",
        "mfi_14",
        "rolling_vol_20",
        "vol_regime"
    ]

    for col in feature_cols:
        df[col] = rolling_zscore(df[col], LOOKBACK_Z)

    df["Ticker"] = ticker

    df = df.dropna()

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_all_raw_data():

    os.makedirs("data/processed", exist_ok=True)

    raw_files = [f for f in os.listdir("data") if f.endswith("_raw.csv")]

    print(f"\nProcessing {len(raw_files)} stocks...\n")

    all_stocks = []

    for file in raw_files:
        try:
            symbol = file.replace("_raw.csv", "")
            print(f"Processing {symbol}...", end=" ", flush=True)

            df = pd.read_csv(
                f"data/{file}",
                header=[0, 1],
                index_col=0
            )

            df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["Close", "High", "Low", "Volume"])

            features = engineer_features(df, symbol)

            if len(features) < 300:
                print("Skipped (too few rows)")
                continue

            all_stocks.append(features)
            print(f"OK ({len(features)} rows)")

        except Exception as e:
            print(f"Error: {e}")

    if len(all_stocks) == 0:
        raise ValueError("No valid stock data found.")

    # ============================================================
    # CROSS-SECTIONAL TARGET + RANKING
    # ============================================================

    print("\nApplying cross-sectional target + ranking...")

    combined = pd.concat(all_stocks)
    combined = combined.sort_index()

    # Cross-sectional target (THIS IS THE CRITICAL FIX)
    combined["target"] = combined.groupby(combined.index)["fwd_return"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    # Cross-sectional feature ranking
    base_features = [
        "ret_5",
        "ma_dist_20",
        "mfi_14",
        "rolling_vol_20",
        "vol_regime"
    ]

    for col in base_features:
        combined[col + "_rank"] = (
            combined.groupby(combined.index)[col]
            .rank(pct=True)
        )

    combined = combined.dropna()

    combined = combined[FEATURE_ORDER + ["target", "Ticker"]]

    combined.to_csv("data/processed/all_stocks_features.csv")

    print("\nFeature engineering completed successfully.\n")


if __name__ == "__main__":
    process_all_raw_data()
