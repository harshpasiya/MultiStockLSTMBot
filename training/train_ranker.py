import os, sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_PATH   = ROOT / "data"    / "processed" / "all_stocks_features.csv"
PRICE_DIR   = ROOT / "data"
MODEL_PATH  = ROOT / "models"  / "zodic_setup_classifier.pkl"
SETUP_CSV   = ROOT / "data"    / "processed" / "setup_labels.csv"
PREDICT_CSV = ROOT / "outputs" / "future_predictions.csv"

HORIZON        = 5
TARGET_RETURN  = 0.010
TRAIN_CUTOFF   = "2025-04-01"
VAL_CUTOFF     = "2025-09-01"
BUY_THRESHOLD  = 0.52

# RAW features — actual values, NOT z-scored
FEATURE_COLS = [
    "setup_type_enc",        # 0=breakout  1=macd_cross  2=pullback
    "strength",              # 0.0–1.0  setup quality
    "rsi_raw",               # 0–100     actual RSI
    "ret_5_raw",             # actual 5-day return
    "ret_20_raw",            # actual 20-day return
    "momentum_skip_raw",     # 60d return skipping last 5d
    "dist_ma20_raw",         # % distance from 20-day MA
    "dist_ma50_raw",         # % distance from 50-day MA
    "adx_raw",               # 0–100  trend strength
    "atr_pct_raw",           # ATR as % of price
    "vol_spike_raw",         # volume / 20d avg volume
    "dist_52w_high_raw",     # % below 52-week high
    "candle_strength_raw",   # (close–open)/(high–low)
    "market_breadth",        # % of universe above MA50 (regime)
    "day_of_week",           # 0=Mon  4=Fri
]


class PlattCalibrated:
    def __init__(self, base_model, X_cal, y_cal):
        raw        = base_model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        self.platt = LogisticRegression(C=1.0, max_iter=500)
        self.platt.fit(raw, y_cal)
        self.base  = base_model
        self.feature_importances_ = getattr(base_model, "feature_importances_", None)

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.platt.predict_proba(raw)


def load_price_data() -> dict:
    import glob
    price_data = {}
    for path in glob.glob(str(PRICE_DIR / "*_NS_raw.csv")):
        ticker     = os.path.basename(path).replace("_NS_raw.csv", "")
        df         = pd.read_csv(path, skiprows=2)
        df.columns = ["Date","Close","High","Low","Open","Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df         = df.sort_values("Date").reset_index(drop=True)
        price_data[ticker] = df
    print(f"  Loaded {len(price_data)} price files")
    return price_data


def compute_raw_features(df_sl: pd.DataFrame) -> dict:
    c, h, l, v, o = (df_sl["Close"], df_sl["High"],
                     df_sl["Low"], df_sl["Volume"], df_sl["Open"])
    # RSI
    delta  = c.diff()
    gain   = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss   = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    rsi    = float((100 - 100 / (1 + gain / (loss + 1e-8))).iloc[-1])
    # Returns
    ret_5  = float(c.pct_change(5).iloc[-1])  if len(c) >= 6  else 0.0
    ret_20 = float(c.pct_change(20).iloc[-1]) if len(c) >= 21 else 0.0
    mom_sk = float(c.shift(5).pct_change(55).iloc[-1]) if len(c) >= 61 else 0.0
    # MA distances
    ma20   = c.rolling(20).mean().iloc[-1]
    ma50   = c.rolling(50).mean().iloc[-1]
    close  = c.iloc[-1]
    d_ma20 = (close - ma20) / (ma20 + 1e-8) if not np.isnan(ma20) else 0.0
    d_ma50 = (close - ma50) / (ma50 + 1e-8) if not np.isnan(ma50) else 0.0
    # ADX
    tr   = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    up_  = h - h.shift(1)
    dn_  = l.shift(1) - l
    pdm  = pd.Series(np.where((up_ > dn_) & (up_ > 0), up_, 0.0), index=h.index)
    mdm  = pd.Series(np.where((dn_ > up_) & (dn_ > 0), dn_, 0.0), index=h.index)
    atr  = tr.ewm(alpha=1/14, min_periods=14).mean()
    pdi  = 100 * pdm.ewm(alpha=1/14, min_periods=14).mean() / (atr + 1e-8)
    mdi  = 100 * mdm.ewm(alpha=1/14, min_periods=14).mean() / (atr + 1e-8)
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-8)
    adx  = float(dx.ewm(alpha=1/14, min_periods=14).mean().iloc[-1])
    # ATR%, vol spike, 52w dist, candle
    atr_pct = float(atr.iloc[-1] / (close + 1e-8))
    vol_sp  = float(v.iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-8))
    h52     = h.rolling(252, min_periods=60).max().iloc[-1]
    d52     = float((close - h52) / (h52 + 1e-8)) if not np.isnan(h52) else 0.0
    cndl    = float((c.iloc[-1] - o.iloc[-1]) / (h.iloc[-1] - l.iloc[-1] + 1e-8))
    return {
        "rsi_raw": rsi, "ret_5_raw": ret_5,
        "ret_20_raw": ret_20, "momentum_skip_raw": mom_sk,
        "dist_ma20_raw": d_ma20, "dist_ma50_raw": d_ma50,
        "adx_raw": adx, "atr_pct_raw": atr_pct,
        "vol_spike_raw": vol_sp, "dist_52w_high_raw": d52,
        "candle_strength_raw": cndl,
    }


def compute_market_breadth(price_data: dict, date: pd.Timestamp) -> float:
    above, total = 0, 0
    for df in price_data.values():
        sl = df[df["Date"] <= date].tail(60)
        if len(sl) < 50:
            continue
        ma50  = sl["Close"].rolling(50).mean().iloc[-1]
        close = sl["Close"].iloc[-1]
        if not np.isnan(ma50):
            above += int(close > ma50)
            total += 1
    return float(above / total) if total > 0 else 0.5


def build_setup_labels(price_data: dict) -> pd.DataFrame:
    from strategies.entry_rules import EntryRules
    rules     = EntryRules()
    rows      = []
    all_dates = sorted(set(
        d for df in price_data.values()
        for d in df["Date"].values
        if pd.Timestamp(d) >= pd.Timestamp("2022-06-01")
    ))
    print(f"  Scanning {len(all_dates)} dates x {len(price_data)} tickers...")
    for i, date in enumerate(all_dates):
        date = pd.Timestamp(date)
        if i % 200 == 0:
            print(f"    {i:>4}/{len(all_dates)} dates  |  setups: {len(rows)}")
        breadth = compute_market_breadth(price_data, date)
        dow     = date.dayofweek
        for ticker, df in price_data.items():
            df_sl = df[df["Date"] <= date].tail(120).copy()
            if len(df_sl) < 60:
                continue
            sig = rules.scan(df_sl, ticker)
            if sig is None:
                continue
            fut = df[df["Date"] > date].head(HORIZON)
            if len(fut) < HORIZON:
                continue
            fwd_ret   = fut["Close"].iloc[-1] / df_sl["Close"].iloc[-1] - 1
            raw_feats = compute_raw_features(df_sl)
            if any(np.isnan(v) for v in raw_feats.values()):
                continue
            rows.append({
                "Date": date, "Ticker": ticker,
                "setup_type": sig.setup_type, "strength": sig.strength,
                "market_breadth": breadth, "day_of_week": dow,
                "fwd_return": fwd_ret, "label": int(fwd_ret > TARGET_RETURN),
                **raw_feats,
            })
    df_out = pd.DataFrame(rows)
    print(f"  Setups: {len(df_out):,}  |  Positive: {df_out['label'].mean()*100:.1f}%")
    for k, v in df_out["setup_type"].value_counts().items():
        print(f"    {k}: {v:,}")
    return df_out


def prepare_features(df: pd.DataFrame, le: LabelEncoder = None):
    df = df.copy()
    if le is None:
        le = LabelEncoder()
        df["setup_type_enc"] = le.fit_transform(df["setup_type"].astype(str))
    else:
        df["setup_type_enc"] = le.transform(df["setup_type"].astype(str))
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["label"].astype(int)
    return X, y, le


def train():
    print("=" * 60)
    print("  ZODIC Setup Classifier v2  —  Raw Features + Breadth")
    print("=" * 60)

    print("\n[1/4] Loading raw price data...")
    price_data = load_price_data()

    if SETUP_CSV.exists():
        print("\n[2/4] Checking cached setup labels...")
        setups = pd.read_csv(SETUP_CSV, parse_dates=["Date"])
        if "rsi_raw" not in setups.columns:
            print("  Old cache detected (z-scored). Rebuilding with raw features...")
            SETUP_CSV.unlink()
            setups = build_setup_labels(price_data)
            setups.to_csv(SETUP_CSV, index=False)
        else:
            print(f"  {len(setups):,} setups loaded  |  raw features confirmed")
    else:
        print("\n[2/4] Building setup labels (~15 min first run)...")
        setups = build_setup_labels(price_data)
        os.makedirs(SETUP_CSV.parent, exist_ok=True)
        setups.to_csv(SETUP_CSV, index=False)

    print(f"  Positive rate: {setups['label'].mean()*100:.1f}%")

    print("\n[3/4] Splitting...")
    train_df = setups[setups["Date"] <  TRAIN_CUTOFF]
    val_df   = setups[(setups["Date"] >= TRAIN_CUTOFF) & (setups["Date"] < VAL_CUTOFF)]
    test_df  = setups[setups["Date"] >= VAL_CUTOFF]
    print(f"  Train: {len(train_df):>5,}  (+{train_df['label'].mean()*100:.1f}%)")
    print(f"  Val  : {len(val_df):>5,}  (+{val_df['label'].mean()*100:.1f}%)")
    print(f"  Test : {len(test_df):>5,}  (+{test_df['label'].mean()*100:.1f}%)")

    X_train, y_train, le = prepare_features(train_df)
    X_val,   y_val,   _  = prepare_features(val_df,  le)
    X_test,  y_test,  _  = prepare_features(test_df, le)

    print("\n[4/4] Training XGBoost with raw features...")
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"  scale_pos_weight: {pos_weight:.2f}")
    try:
        from xgboost import XGBClassifier
        base = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.80, colsample_bytree=0.80, min_child_weight=8,
            reg_alpha=0.05, reg_lambda=1.0, scale_pos_weight=pos_weight,
            eval_metric="auc", random_state=42, verbosity=0,
        )
        base.fit(X_train, y_train)
        print(f"  Trained {base.n_estimators} trees")
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        base = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.80, min_samples_leaf=15, random_state=42,
        )
        base.fit(X_train, y_train)

    print("  Calibrating (Platt scaling)...")
    model = PlattCalibrated(base, X_val, y_val)

    print(f"\n{'='*60}")
    print(f"  EVALUATION  (threshold={BUY_THRESHOLD})")
    print(f"{'='*60}")
    for name, Xs, ys in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        probs = model.predict_proba(Xs)[:, 1]
        preds = (probs > BUY_THRESHOLD).astype(int)
        auc   = roc_auc_score(ys, probs)
        prec  = precision_score(ys, preds, zero_division=0)
        rec   = recall_score(ys, preds, zero_division=0)
        f1    = f1_score(ys, preds, zero_division=0)
        print(f"\n  {name}  (n={len(ys):,}):")
        print(f"    AUC       : {auc:.4f}   [target > 0.58]")
        print(f"    Precision : {prec:.4f}   [target > 0.58]")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1        : {f1:.4f}")
        print(f"    BUY calls : {preds.sum()} / {len(ys)}")

    if model.feature_importances_ is not None:
        fi = pd.Series(model.feature_importances_,
                       index=FEATURE_COLS).sort_values(ascending=False)
        print("\n  Feature Importance (top 10):")
        for feat, imp in fi.head(10).items():
            print(f"    {feat:<22} {imp:.4f}  {'█'*int(imp*80)}")

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump({"model": model, "le": le,
                 "features": FEATURE_COLS, "threshold": BUY_THRESHOLD}, MODEL_PATH)
    print(f"\n  Model saved -> {MODEL_PATH}")
    save_predictions(model, le, price_data)


def save_predictions(model, le, price_data: dict):
    from strategies.entry_rules import EntryRules
    rules       = EntryRules()
    latest_date = max(df["Date"].max() for df in price_data.values())
    breadth     = compute_market_breadth(price_data, latest_date)
    dow         = latest_date.dayofweek

    print(f"\n{'='*60}")
    print(f"  LIVE PREDICTIONS  —  {latest_date.date()}")
    print(f"  Market breadth   :  {breadth*100:.1f}% stocks above MA50")
    print(f"{'='*60}")

    results = []
    for ticker, df in price_data.items():
        df_sl = df[df["Date"] <= latest_date].tail(120).copy()
        if len(df_sl) < 60:
            continue
        sig = rules.scan(df_sl, ticker)
        if sig is None:
            continue
        raw = compute_raw_features(df_sl)
        if any(np.isnan(v) for v in raw.values()):
            continue
        row = {"setup_type": sig.setup_type, "strength": sig.strength,
               "market_breadth": breadth, "day_of_week": dow, **raw}
        row["setup_type_enc"] = le.transform([sig.setup_type])[0]
        X    = pd.DataFrame([row])[FEATURE_COLS]
        X    = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prob = float(model.predict_proba(X)[0, 1])
        results.append({
            "Ticker": ticker, "Setup": sig.setup_type,
            "Strength": sig.strength, "ML_Prob": round(prob, 4),
            "Breadth": round(breadth, 3), "StopLoss": sig.stop_loss,
            "Signal": "BUY" if prob > BUY_THRESHOLD else "NEUTRAL",
            "Notes": sig.notes,
        })

    if not results:
        print("  No TA setups fired today.")
        return
    df_out = (pd.DataFrame(results)
                .sort_values("ML_Prob", ascending=False)
                .reset_index(drop=True))
    os.makedirs(PREDICT_CSV.parent, exist_ok=True)
    df_out.to_csv(PREDICT_CSV, index=False)

    buy = df_out[df_out["Signal"] == "BUY"]
    print(f"  BUY signals (TA + ML > {BUY_THRESHOLD}): {len(buy)} / {len(df_out)}")
    if len(buy):
        print(buy[["Ticker","Setup","Strength","ML_Prob","StopLoss"]].to_string(index=False))
    else:
        print("  No setups pass filter — stay flat.")
    print(f"\n  Saved -> {PREDICT_CSV}")


if __name__ == "__main__":
    os.makedirs(ROOT / "models",  exist_ok=True)
    os.makedirs(ROOT / "outputs", exist_ok=True)
    train()
