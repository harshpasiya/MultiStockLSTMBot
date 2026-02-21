import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = "data/processed/all_stocks_features.csv"
HORIZON = 5


# ============================================================
# PER STOCK FEATURES
# ============================================================

def compute_stock_features(path):

    ticker = os.path.basename(path).replace("_NS_raw.csv", "")

    df = pd.read_csv(path, skiprows=2)
    df.columns = ["Date","Close","High","Low","Open","Volume"]

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    o,h,l,c,v = df["Open"],df["High"],df["Low"],df["Close"],df["Volume"]

    ret1 = c.pct_change()
    ret2 = c.pct_change(2)
    ret5 = c.pct_change(5)
    ret20 = c.pct_change(20)
    ret60 = c.pct_change(60)

    # ----- Momentum -----
    df["ret_5"] = ret5
    df["ret_20"] = ret20
    df["ret_60"] = ret60
    df["momentum_accel"] = ret5 - ret20
    df["trend_stability"] = ret1.rolling(10).mean()/(ret1.rolling(10).std()+1e-8)

    # ----- Mean Reversion -----
    df["short_reversal"] = -ret2
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["price_zscore_20"] = (c-ma20)/(std20+1e-8)
    df["intraday_reversal"] = (c-o)/((h-l)+1e-8)

    # ----- Volatility -----
    std5 = ret1.rolling(5).std()
    std20 = ret1.rolling(20).std()
    df["vol_expansion"] = std5/(std20+1e-8)

    atr20 = (h-l).rolling(20).mean()
    df["range_expansion"] = (h-l)/(atr20+1e-8)
    df["gap_pressure"] = (o-c.shift(1))/(c.shift(1)+1e-8)

    # ----- Liquidity -----
    df["volume_shock"] = v/(v.rolling(20).mean()+1e-8)
    df["dollar_volume"] = c*v



    # target
    raw_fwd = c.shift(-HORIZON) / c - 1
    df["raw_target"] = raw_fwd

    df["Ticker"] = ticker

    return df


# ============================================================
# MARKET CONTEXT FEATURES (NEW CORE)
# ============================================================

def add_market_context(df):

    context_rows = []

    for date, day in df.groupby("Date"):

        d = {}

        d["Date"] = date

        # momentum structure
        d["cs_momentum_median"] = day["ret_20"].median()
        d["cs_momentum_dispersion"] = day["ret_20"].std()
        d["cs_market_return"] = day["raw_target"].mean()
        # volatility disagreement
        d["cs_vol_dispersion"] = day["vol_expansion"].std()

        # leadership clarity
        winners = day["ret_5"].quantile(0.9)
        losers = day["ret_5"].quantile(0.1)
        d["cs_winner_loser_spread"] = winners - losers

        # capital concentration
        dv = day["dollar_volume"]
        d["cs_volume_concentration"] = dv.max()/(dv.sum()+1e-8)

        # breadth
        d["cs_trend_breadth"] = (day["ret_20"] > 0).mean()

        context_rows.append(d)

    context = pd.DataFrame(context_rows)

    df = df.merge(context, on="Date", how="left")

    return df


# ============================================================
# CROSS SECTIONAL RANK NORMALIZATION
# ============================================================

def rank_normalize(df):

    # -----------------------------
    # 1) MARKET NEUTRAL TARGET
    # -----------------------------
    df["target"] = df["raw_target"] - df["cs_market_return"]

    # -----------------------------
    # 2) VOLATILITY NORMALIZATION
    # -----------------------------
    vol = (
        df.groupby("Ticker")["raw_target"]
        .rolling(60)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["target"] = df["target"] / (vol + 1e-6)

    # drop unstable early rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])

    # -----------------------------
    # 3) CROSS-SECTION RANK FEATURES
    # -----------------------------
    stock_cols = [
        "ret_20", "ret_60", "momentum_accel", "trend_stability",
        "short_reversal", "price_zscore_20", "intraday_reversal",
        "vol_expansion", "range_expansion", "gap_pressure",
        "volume_shock", "dollar_volume",
        "rel_mom", "vol_edge", "crowding", "mom_shift_5"
    ]

    context_cols = [
        "cs_momentum_median","cs_momentum_dispersion","cs_vol_dispersion",
        "cs_winner_loser_spread","cs_volume_concentration","cs_trend_breadth"
    ]

    ranked_days = []

    for date, day in df.groupby("Date"):

        day = day.copy()

        # -------------------------------------------------
        # NEW: RELATIVE POSITION FEATURES (FIRST)
        # -------------------------------------------------

        # relative momentum vs universe median
        day["rel_mom"] = day["ret_20"] - day["ret_20"].median()

        # volatility advantage vs peers
        day["vol_edge"] = day["vol_expansion"] - day["vol_expansion"].median()

        # crowding: extremes persist more than middle
        rank_tmp = day["ret_20"].rank(pct=True)
        day["crowding"] = (rank_tmp - 0.5).abs()

        # percentile shift (leader acceleration)
        day = day.sort_values("Ticker")
        day["mom_rank"] = rank_tmp.values
        day["mom_shift_5"] = day.groupby("Ticker")["mom_rank"].diff(5)

        # -------------------------------------------------
        # NOW rank features
        # -------------------------------------------------
        for col in stock_cols:
            day[col + "_rank"] = day[col].rank(pct=True)

        # z-score context features across history
        for col in context_cols:
            mean = df[col].mean()
            std = df[col].std() + 1e-8
            day[col+"_rank"] = (day[col] - mean) / std

        # -------------------------------------------------
        # NEW: RELATIVE POSITION FEATURES (CRITICAL)
        # -------------------------------------------------

        # relative momentum vs universe median
        day["rel_mom"] = day["ret_20"] - day["ret_20"].median()

        # volatility advantage vs peers
        day["vol_edge"] = day["vol_expansion"] - day["vol_expansion"].median()

        # crowding: extremes persist more than middle
        rank_tmp = day["ret_20"].rank(pct=True)
        day["crowding"] = (rank_tmp - 0.5).abs()

        # percentile shift (leader acceleration)
        day = day.sort_values("Ticker")
        day["mom_rank"] = rank_tmp.values
        day["mom_shift_5"] = day.groupby("Ticker")["mom_rank"].diff(5)
        # FINAL TARGET RANK (AFTER neutralization)
        day["target"] = day["target"].rank(pct=True) - 0.5

        ranked_days.append(day)

    df = pd.concat(ranked_days)
    # keep rows where target exists
    df = df.dropna(subset=["target"])

    # forward fill features within each stock
    feature_cols = [c for c in df.columns if c.endswith("_rank")]
    df[feature_cols] = df.groupby("Ticker")[feature_cols].ffill()

    # remaining NaN → neutral value (0.5 rank)
    df[feature_cols] = df[feature_cols].fillna(0.5)
    keep = ["Date","Ticker","target"] + [c for c in df.columns if c.endswith("_rank")]
    df = df[keep].sort_values(["Date","Ticker"])

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_dataset():

    files = glob.glob(os.path.join(DATA_DIR,"*_NS_raw.csv"))

    all_data = []
    for f in files:
        print("Processing",os.path.basename(f))
        all_data.append(compute_stock_features(f))

    df = pd.concat(all_data).dropna()

    print("Adding market context...")
    df = add_market_context(df)

    print("Ranking normalization...")
    df = rank_normalize(df)

    os.makedirs("data/processed",exist_ok=True)
    df.to_csv(OUTPUT_FILE,index=False)

    print("\nSaved →",OUTPUT_FILE)


if __name__ == "__main__":
    build_dataset()