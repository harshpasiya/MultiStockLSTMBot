import os, glob
import numpy as np
import pandas as pd

DATA_DIR         = "data"
OUTPUT_FILE      = "data/processed/all_stocks_features.csv"
HORIZON          = 5
TARGET_THRESHOLD = 0.010   # +1.0% in 5 days = positive label


def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_stock_features(path):
    ticker     = os.path.basename(path).replace("_NS_raw.csv", "")
    df         = pd.read_csv(path, skiprows=2)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df         = df.sort_values("Date").reset_index(drop=True)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    ret1  = c.pct_change()
    ret2  = c.pct_change(2)
    ret5  = c.pct_change(5)
    ret20 = c.pct_change(20)
    ret60 = c.pct_change(60)

    df["ret_5"]           = ret5
    df["ret_20"]          = ret20
    df["ret_60"]          = ret60
    df["momentum_accel"]  = ret5 - ret20
    df["trend_stability"] = ret1.rolling(10).mean() / (ret1.rolling(10).std() + 1e-8)
    df["short_reversal"]  = -ret2
    ma20                  = c.rolling(20).mean()
    std20_c               = c.rolling(20).std()
    df["price_zscore_20"] = (c - ma20) / (std20_c + 1e-8)
    df["intraday_reversal"] = (c - o) / ((h - l) + 1e-8)
    df["bb_position"]     = (c - (ma20 - 2*std20_c)) / (4*std20_c + 1e-8)
    df["rsi_norm"]        = (compute_rsi(c, 14) - 50) / 50
    std5                  = ret1.rolling(5).std()
    std20                 = ret1.rolling(20).std()
    df["vol_expansion"]   = std5 / (std20 + 1e-8)
    atr20                 = (h - l).rolling(20).mean()
    df["range_expansion"] = (h - l) / (atr20 + 1e-8)
    df["gap_pressure"]    = (o - c.shift(1)) / (c.shift(1) + 1e-8)
    df["volume_shock"]    = v / (v.rolling(20).mean() + 1e-8)
    df["dollar_volume"]   = c * v
    df["raw_target"]      = c.shift(-HORIZON) / c - 1
    df["Ticker"]          = ticker
    return df


def add_market_context(df):
    context_rows = []
    for date, day in df.groupby("Date"):
        d = {"Date": date}
        d["cs_momentum_median"]     = day["ret_20"].median()
        d["cs_momentum_dispersion"] = day["ret_20"].std()
        d["cs_market_return"]       = day["raw_target"].mean()
        d["cs_vol_dispersion"]      = day["vol_expansion"].std()
        d["cs_winner_loser_spread"] = day["ret_5"].quantile(0.9) - day["ret_5"].quantile(0.1)
        dv = day["dollar_volume"]
        d["cs_volume_concentration"] = dv.max() / (dv.sum() + 1e-8)
        d["cs_trend_breadth"]       = (day["ret_20"] > 0).mean()
        context_rows.append(d)
    return df.merge(pd.DataFrame(context_rows), on="Date", how="left")


def rank_normalize(df):
    # ── BINARY TARGET ─────────────────────────────────────────────────────────
    # 1 = stock will return > +1.0% in 5 days  (absolute, not relative)
    # 0 = otherwise
    # Model learns ABSOLUTE upside potential, not "best among bad stocks"
    df["target"] = (df["raw_target"] > TARGET_THRESHOLD).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])

    base_stock_cols = [
        "ret_5", "ret_20", "ret_60", "momentum_accel", "trend_stability",
        "short_reversal", "price_zscore_20", "intraday_reversal",
        "bb_position", "rsi_norm", "vol_expansion", "range_expansion",
        "gap_pressure", "volume_shock", "dollar_volume",
    ]
    context_cols = [
        "cs_momentum_median", "cs_momentum_dispersion", "cs_vol_dispersion",
        "cs_winner_loser_spread", "cs_volume_concentration", "cs_trend_breadth",
    ]

    ranked_days = []
    for date, day in df.groupby("Date"):
        day = day.copy()
        day["rel_mom"]  = day["ret_20"] - day["ret_20"].median()
        day["vol_edge"] = day["vol_expansion"] - day["vol_expansion"].median()
        day["crowding"] = (day["ret_20"].rank(pct=True) - 0.5).abs()
        for col in base_stock_cols:
            if col in day.columns:
                day[col + "_rank"] = day[col].rank(pct=True)
        day["rel_mom_rank"]  = day["rel_mom"].rank(pct=True)
        day["vol_edge_rank"] = day["vol_edge"].rank(pct=True)
        day["crowding_rank"] = day["crowding"].rank(pct=True)
        for col in context_cols:
            mean = df[col].mean();  std = df[col].std() + 1e-8
            day[col + "_rank"] = (day[col] - mean) / std
        ranked_days.append(day)

    df = pd.concat(ranked_days).dropna(subset=["target"])
    df = df.sort_values(["Ticker", "Date"])
    df["mom_shift_5_rank"] = df.groupby("Ticker")["ret_20_rank"].diff(5)
    df["mom_shift_5_rank"] = df.groupby("Date")["mom_shift_5_rank"].transform(
        lambda x: x.rank(pct=True)
    )
    feature_cols = [c for c in df.columns if c.endswith("_rank")]
    df[feature_cols] = df.groupby("Ticker")[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].fillna(0.5)

    SELECTED_FEATURES = [
        "ret_5_rank", "ret_20_rank", "ret_60_rank", "momentum_accel_rank",
        "trend_stability_rank", "short_reversal_rank", "price_zscore_20_rank",
        "rsi_norm_rank", "vol_expansion_rank", "volume_shock_rank",
        "rel_mom_rank", "crowding_rank", "mom_shift_5_rank",
    ]
    pos_rate = df["target"].mean()
    print(f"  Label distribution: {pos_rate*100:.1f}% positive  "
          f"{(1-pos_rate)*100:.1f}% negative")
    keep = ["Date", "Ticker", "target"] + SELECTED_FEATURES
    return df.sort_values(["Date", "Ticker"])[keep]


def build_dataset():
    files = glob.glob(os.path.join(DATA_DIR, "*_NS_raw.csv"))
    if not files:
        raise FileNotFoundError(f"No *_NS_raw.csv found in {DATA_DIR}/")
    all_data = []
    for f in files:
        print("Processing", os.path.basename(f))
        all_data.append(compute_stock_features(f))
    df = pd.concat(all_data).dropna(subset=["Close"])
    print(f"\nRows: {len(df)} | Stocks: {df['Ticker'].nunique()}")
    df = add_market_context(df)
    df = rank_normalize(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved -> {OUTPUT_FILE} | Rows: {len(df)}")

if __name__ == "__main__":
    build_dataset()
