import os, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

DATA_DIR    = "data"
OUTPUT_FILE = "data/processed/all_stocks_features.csv"
HORIZON     = 5
ZSCORE_WIN  = 252       # 1 trading year
ZSCORE_MIN  = 60        # minimum history needed
CLIP        = 3.0       # z-score clip

FEATURE_COLS = [
    "ret_5", "ret_20", "momentum_skip",       # ① Momentum
    "rsi_14", "bb_pct", "dist_ma50",          # ② Mean Reversion
    "adx_14", "macd_hist", "trend_slope",     # ③ Trend Quality
    "atr_pct", "vol_ratio",                   # ④ Volatility
    "vol_spike", "obv_slope",                 # ⑤ Volume / Flow
    "dist_52w_high", "candle_strength",       # ⑥ Relative / Structure
    "rel_strength",                           # ⑥ Alpha (added cross-sectionally)
]
TARGET_COLS = [f"target_d{n}" for n in range(1, HORIZON + 1)]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def rolling_zscore(s, win=ZSCORE_WIN, minp=ZSCORE_MIN, clip=CLIP):
    """Normalize feature vs this stock's own history. Clips to ±3."""
    mu  = s.rolling(win, min_periods=minp).mean()
    sig = s.rolling(win, min_periods=minp).std()
    return ((s - mu) / (sig + 1e-8)).clip(-clip, clip)


def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-8)))


def compute_adx(high, low, close, period=14):
    """
    Wilder's ADX — trend STRENGTH (0-100).
    > 25 = strong trend, < 20 = choppy.
    Institutions avoid trading when ADX < 20.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up, down     = high - high.shift(1), low.shift(1) - low
    plus_dm      = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm     = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr          = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di      = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-8)
    minus_di     = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-8)
    dx           = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    return dx.ewm(alpha=1/period, min_periods=period).mean()


def compute_macd_hist(close, atr, fast=12, slow=26, signal=9):
    """
    MACD histogram, ATR-normalised.
    Growing positive = momentum accelerating up.
    Crossing zero up = early trend start signal.
    """
    ema_f = close.ewm(span=fast,   min_periods=fast).mean()
    ema_s = close.ewm(span=slow,   min_periods=slow).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal,  min_periods=signal).mean()
    return (macd - sig) / (atr + 1e-8)


def compute_obv_slope(close, volume, period=14):
    """
    OBV slope — institutional accumulation/distribution detector.
    Rising OBV with rising price = institutions buying.
    Divergence = smart money exiting.
    """
    obv      = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    obv_norm = obv / (obv.rolling(ZSCORE_WIN, min_periods=ZSCORE_MIN).median().abs() + 1)
    return (obv_norm - obv_norm.shift(period)) / (period + 1e-8)


# ── PER-STOCK COMPUTATION ─────────────────────────────────────────────────────

def compute_features_per_stock(path: str) -> pd.DataFrame:
    ticker     = os.path.basename(path).replace("_NS_raw.csv", "")
    df         = pd.read_csv(path, skiprows=2)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df         = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    tr  = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=10).mean()

    # ① Momentum
    ret_5         = c.pct_change(5)
    ret_20        = c.pct_change(20)
    momentum_skip = c.shift(5).pct_change(55)   # 60d return, skip last 5d (Fama-French style)

    # ② Mean Reversion
    rsi_14    = compute_rsi(c, 14)
    ma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    bb_pct    = (c - (ma20 - 2*std20)) / (4*std20 + 1e-8)   # 0=lower band, 1=upper band
    ma50      = c.rolling(50).mean()
    dist_ma50 = (c - ma50) / (ma50 + 1e-8)

    # ③ Trend Quality
    adx_14      = compute_adx(h, l, c, 14)
    macd_hist   = compute_macd_hist(c, atr)
    trend_slope = c.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x / x[0], 1)[0], raw=True
    )

    # ④ Volatility
    atr_pct   = atr / (c + 1e-8)
    vol_ratio = c.pct_change().rolling(5).std() / (c.pct_change().rolling(20).std() + 1e-8)

    # ⑤ Volume / Flow
    vol_spike     = v / (v.rolling(20).mean() + 1e-8)
    obv_slope     = compute_obv_slope(c, v, 14)

    # ⑥ Structure
    high_52w      = h.rolling(252, min_periods=60).max()
    dist_52w_high = (c - high_52w) / (high_52w + 1e-8)   # 0=at 52w high, -0.3=30% below
    candle_strength = (c - o) / (h - l + 1e-8)           # body vs range (institutional conviction)

    raw = pd.DataFrame({
        "Date": df["Date"], "Ticker": ticker,
        "ret_5": ret_5, "ret_20": ret_20, "momentum_skip": momentum_skip,
        "rsi_14": rsi_14, "bb_pct": bb_pct, "dist_ma50": dist_ma50,
        "adx_14": adx_14, "macd_hist": macd_hist, "trend_slope": trend_slope,
        "atr_pct": atr_pct, "vol_ratio": vol_ratio,
        "vol_spike": vol_spike, "obv_slope": obv_slope,
        "dist_52w_high": dist_52w_high, "candle_strength": candle_strength,
    })

    # ── TARGET: 5-day cumulative return PATH ──────────────────────────
    # target_dN = (close at t+N / close at t) - 1
    # Decoder predicts the full trajectory, not just the endpoint.
    for n in range(1, HORIZON + 1):
        raw[f"target_d{n}"] = c.shift(-n) / c - 1

    # ── ROLLING Z-SCORE NORMALISATION (per stock, vs own history) ─────
    for col in FEATURE_COLS[:-1]:       # rel_strength added later
        if col in raw.columns:
            raw[col] = rolling_zscore(raw[col])

    return raw


# ── CROSS-SECTIONAL RELATIVE STRENGTH ────────────────────────────────────────

def add_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    rel_strength = stock's z-scored ret_20 MINUS universe daily median.
    Pure alpha signal: is this stock stronger than the average stock today?
    """
    day_med           = df.groupby("Date")["ret_20"].transform("median")
    df["rel_strength"]= df["ret_20"] - day_med
    df["rel_strength"]= df.groupby("Ticker")["rel_strength"].transform(rolling_zscore)
    return df


# ── BUILD ─────────────────────────────────────────────────────────────────────

def build_dataset():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_NS_raw.csv")))
    if not files:
        raise FileNotFoundError(f"No *_NS_raw.csv in {DATA_DIR}/")

    print(f"Processing {len(files)} stocks...")
    frames = []
    for path in files:
        ticker = os.path.basename(path).replace("_NS_raw.csv", "")
        try:
            frames.append(compute_features_per_stock(path))
            print(f"  ✓  {ticker}")
        except Exception as e:
            print(f"  ✗  {ticker}  SKIPPED ({e})")

    df = pd.concat(frames, ignore_index=True)
    print("Adding relative strength (cross-sectional)...")
    df = add_relative_strength(df)

    keep = ["Date", "Ticker"] + FEATURE_COLS + TARGET_COLS
    df   = df[keep].replace([np.inf, -np.inf], np.nan)
    df   = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    df   = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*55}")
    print(f"  FEATURE ENGINEERING COMPLETE")
    print(f"{'='*55}")
    print(f"  Stocks     : {df['Ticker'].nunique()}")
    print(f"  Total rows : {len(df):,}")
    print(f"  Features   : {len(FEATURE_COLS)}  (institutional grade)")
    print(f"  Targets    : {len(TARGET_COLS)}  (5-day path d1..d5)")
    print(f"  Date range : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"\n  Features:")
    groups = [
        ("① Momentum",       ["ret_5", "ret_20", "momentum_skip"]),
        ("② Mean Reversion", ["rsi_14", "bb_pct", "dist_ma50"]),
        ("③ Trend Quality",  ["adx_14", "macd_hist", "trend_slope"]),
        ("④ Volatility",     ["atr_pct", "vol_ratio"]),
        ("⑤ Volume/Flow",    ["vol_spike", "obv_slope"]),
        ("⑥ Relative",       ["dist_52w_high", "candle_strength", "rel_strength"]),
    ]
    for group, cols in groups:
        print(f"    {group}: {', '.join(cols)}")
    print(f"\n  Targets: {TARGET_COLS}")
    print(f"  Saved  : {OUTPUT_FILE}")
    print(f"{'='*55}")


if __name__ == "__main__":
    build_dataset()
