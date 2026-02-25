import os, glob
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

ROOT        = Path(__file__).resolve().parents[1]
PRICE_DIR   = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
REPORT_CSV  = OUTPUT_DIR / "backtest_trades.csv"
EQUITY_CSV  = OUTPUT_DIR / "equity_curve.csv"
SUMMARY_TXT = OUTPUT_DIR / "backtest_summary.txt"

# ── CONFIGURATION ──────────────────────────────────────────────────────
INITIAL_CAPITAL    = 500_000
CAPITAL_PER_TRADE  = 0.33
MAX_OPEN_POSITIONS = 3
MAX_TRADES_PM      = 8
MAX_HOLD_DAYS      = 15
BREADTH_MIN        = 0.40
START_DATE         = "2022-06-01"
END_DATE           = "2026-02-20"
ENTRY_PRICE_MIN    = 50.0
RISK_PER_TRADE_PCT = 0.012
TRAILING_TRIGGER   = 0.040
TRAILING_STOP_PCT  = 0.020

NIFTY50_EXCLUDE = {
    "ADANIENT",  "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV", "BAJFINANCE", "BEL",        "BHARTIARTL",
    "BPCL",      "BRITANNIA",  "CIPLA",      "COALINDIA",  "DRREDDY",
    "EICHERMOT", "GRASIM",     "HCLTECH",    "HDFCBANK",   "HDFCLIFE",
    "HEROMOTOCO","HINDALCO",   "HINDUNILVR", "ICICIBANK",  "INDUSINDBK",
    "INFY",      "ITC",        "JIOFIN",     "JSWSTEEL",   "KOTAKBANK",
    "LT",        "LTIM",       "M&M",        "MARUTI",     "NESTLEIND",
    "NTPC",      "ONGC",       "POWERGRID",  "RELIANCE",   "SBILIFE",
    "SBIN",      "SHRIRAMFIN", "SUNPHARMA",  "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TCS",        "TECHM",      "TITAN",      "ULTRACEMCO",
    "WIPRO",
}


@dataclass
class Position:
    ticker          : str
    entry_date      : object
    entry_price     : float
    signal_close    : float
    signal_low      : float
    shares          : int
    stop_loss       : float
    target          : float
    sector          : str
    highest_price   : float = 0.0
    trailing_active : bool  = False
    trail_stop      : float = 0.0

@dataclass
class Trade:
    ticker      : str
    entry_date  : object
    exit_date   : object
    entry_price : float
    exit_price  : float
    shares      : int
    pnl         : float
    pnl_pct     : float
    exit_reason : str
    sector      : str


# ── INDICATORS ──────────────────────────────────────────────────────────

def compute_ema(close, span):
    return close.ewm(span=span, min_periods=span).mean()

def compute_rsi(close, period=14):
    d    = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-8))

def compute_atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=10).mean()

def compute_adx(high, low, close, period=14):
    tr  = pd.concat([high - low,
                     (high - close.shift(1)).abs(),
                     (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    up  = high - high.shift(1)
    dn  = low.shift(1) - low
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    ae  = tr.ewm(alpha=1/period, min_periods=period).mean()
    pdi = 100 * pdm.ewm(alpha=1/period, min_periods=period).mean() / (ae + 1e-8)
    mdi = 100 * mdm.ewm(alpha=1/period, min_periods=period).mean() / (ae + 1e-8)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-8)
    return dx.ewm(alpha=1/period, min_periods=period).mean()


# ── DATA ────────────────────────────────────────────────────────────────

def load_all_prices():
    price_data = {}
    for path in glob.glob(str(PRICE_DIR / "*_NS_raw.csv")):
        ticker = os.path.basename(path).replace("_NS_raw.csv", "")
        if ticker in NIFTY50_EXCLUDE:
            continue
        df = pd.read_csv(path, skiprows=2)
        df.columns = ["Date","Close","High","Low","Open","Volume"]
        df["Date"]   = pd.to_datetime(df["Date"])
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        df = df.sort_values("Date").reset_index(drop=True)
        price_data[ticker] = df
    print(f"  Loaded {len(price_data)} stocks  (Nifty50 excluded)")
    return price_data

def precompute_indicators(price_data):
    inds = {}
    for ticker, df in price_data.items():
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        inds[ticker] = {
            "ema9":  compute_ema(c, 9),
            "ema21": compute_ema(c, 21),
            "ema50": compute_ema(c, 50),
            "rsi14": compute_rsi(c, 14),
            "atr14": compute_atr(h, l, c, 14),
            "adx14": compute_adx(h, l, c, 14),
            "vol20": v.rolling(20).mean(),
        }
    print(f"  Indicators for {len(inds)} stocks")
    return inds


# ── REGIME ──────────────────────────────────────────────────────────────

def get_breadth(price_data, date):
    above, total = 0, 0
    for df in price_data.values():
        sl = df[df["Date"] <= date].tail(55)
        if len(sl) < 50: continue
        ma50 = sl["Close"].rolling(50).mean().iloc[-1]
        if not np.isnan(ma50):
            above += int(sl["Close"].iloc[-1] > ma50)
            total += 1
    if total == 0: return True, 0.5
    return (above / total) >= BREADTH_MIN, above / total


# ── SIGNAL ───────────────────────────────────────────────────────────────
# Fires on EOD close of day D → entry at OPEN of day D+1
#
# C1. EMA9 crossed above EMA21 TODAY (yesterday EMA9 <= EMA21)
# C2. RSI(14) between 50 and 70
# C3. Close > EMA50
# C4. ADX(14) > 18
# C5. Volume >= 20d average
# C6. Green candle (close > open)
# C7. Close > Open by >= 0.5%

def check_signal(ticker, df_sl, ind_sl):
    if len(df_sl) < 30: return False, None
    c, h, l, o, v = (df_sl["Close"], df_sl["High"], df_sl["Low"],
                     df_sl["Open"],  df_sl["Volume"])
    close_d = c.iloc[-1]; open_d = o.iloc[-1]; low_d = l.iloc[-1]

    ema9_t  = ind_sl["ema9"].iloc[-1];  ema9_y  = ind_sl["ema9"].iloc[-2]
    ema21_t = ind_sl["ema21"].iloc[-1]; ema21_y = ind_sl["ema21"].iloc[-2]
    ema50_t = ind_sl["ema50"].iloc[-1]
    rsi_t   = ind_sl["rsi14"].iloc[-1]
    adx_t   = ind_sl["adx14"].iloc[-1]
    vol_t   = v.iloc[-1]; vol20_t = ind_sl["vol20"].iloc[-1]

    if any(pd.isna(x) for x in [ema9_t,ema9_y,ema21_t,ema21_y,
                                  ema50_t,rsi_t,adx_t,vol20_t]):
        return False, None
    if close_d < ENTRY_PRICE_MIN or vol20_t < 1:
        return False, None

    if not ((ema9_y <= ema21_y) and (ema9_t > ema21_t) and   # C1 fresh cross
            50 < rsi_t < 70 and                               # C2 momentum
            close_d > ema50_t and                             # C3 uptrend
            adx_t > 18 and                                    # C4 trending
            vol_t >= vol20_t * 1.0 and                        # C5 volume
            close_d > open_d and                              # C6 green
            (close_d - open_d) / open_d >= 0.005):            # C7 decisive
        return False, None

    quality = adx_t * 0.4 + (vol_t / vol20_t) * 8 + rsi_t * 0.2
    return True, {"signal_close": close_d, "signal_low": low_d,
                  "quality": quality, "sector": "OTHER"}


# ── POSITION SIZING ──────────────────────────────────────────────────────

def size_position(capital, entry_price, stop_price):
    risk_amt     = capital * RISK_PER_TRADE_PCT
    risk_per_sh  = max(entry_price - stop_price, entry_price * 0.005)
    risk_shares  = int(risk_amt / risk_per_sh)
    alloc_shares = int(capital * CAPITAL_PER_TRADE / entry_price)
    return max(1, min(risk_shares, alloc_shares))

def open_value(positions, price_data, date):
    val = 0.0
    for ticker, pos in positions.items():
        sl = price_data[ticker][price_data[ticker]["Date"] <= date]
        if not sl.empty: val += sl["Close"].iloc[-1] * pos.shares
    return val


# ── BACKTEST ENGINE ──────────────────────────────────────────────────────

def run_backtest(price_data, inds):
    capital = float(INITIAL_CAPITAL)
    positions = {}; pending = {}; closed = []; equity_curve = []
    bull_regime = True; month_trades = {}

    all_dates = sorted(set(
        d for df in price_data.values()
        for d in df["Date"].values
        if START_DATE <= str(d)[:10] <= END_DATE
    ))
    print(f"  Period : {str(all_dates[0])[:10]}  to  {str(all_dates[-1])[:10]}")
    print(f"  Days   : {len(all_dates)}")

    for day_idx, date in enumerate(all_dates):
        date = pd.Timestamp(date)
        mkey = date.strftime("%Y-%m")

        # ── FILL PENDING AT TODAY'S OPEN ─────────────────────────────
        filled = []
        for ticker, sig in list(pending.items()):
            if ticker in positions or len(positions) >= MAX_OPEN_POSITIONS:
                filled.append(ticker); continue
            if month_trades.get(mkey, 0) >= MAX_TRADES_PM:
                filled.append(ticker); continue
            today_row = price_data[ticker][price_data[ticker]["Date"] == date]
            if today_row.empty: filled.append(ticker); continue
            entry_p = today_row["Open"].iloc[0]
            if entry_p <= 0 or pd.isna(entry_p): filled.append(ticker); continue
            stop    = round(sig["signal_low"] * 0.997, 2)
            if entry_p <= stop: filled.append(ticker); continue   # gapped down
            risk    = entry_p - stop
            if risk <= 0: filled.append(ticker); continue
            target  = round(entry_p + 2.5 * risk, 2)
            tp = (target - entry_p) / entry_p
            sp = (entry_p - stop)   / entry_p
            if tp < 0.02 or sp > 0.05: filled.append(ticker); continue
            if capital * CAPITAL_PER_TRADE < entry_p: filled.append(ticker); continue
            shares = size_position(capital, entry_p, stop)
            cost   = shares * entry_p
            if cost > capital: filled.append(ticker); continue
            capital -= cost
            positions[ticker] = Position(
                ticker=ticker, entry_date=date, entry_price=entry_p,
                signal_close=sig["signal_close"], signal_low=sig["signal_low"],
                shares=shares, stop_loss=stop, target=target,
                sector=sig["sector"], highest_price=entry_p, trail_stop=stop)
            month_trades[mkey] = month_trades.get(mkey, 0) + 1
            filled.append(ticker)
        for t in filled: pending.pop(t, None)

        # ── EXIT ──────────────────────────────────────────────────────
        to_close = []
        for ticker, pos in positions.items():
            row = price_data[ticker][price_data[ticker]["Date"] == date]
            if row.empty: continue
            hd = row["High"].iloc[0]; ld = row["Low"].iloc[0]
            cd = row["Close"].iloc[0]
            days_held = (date - pos.entry_date).days
            if cd > pos.highest_price: pos.highest_price = cd
            if (pos.highest_price - pos.entry_price) / pos.entry_price >= TRAILING_TRIGGER:
                pos.trailing_active = True
            if pos.trailing_active:
                pos.trail_stop = max(pos.trail_stop,
                                     pos.highest_price * (1 - TRAILING_STOP_PCT))
                pos.stop_loss  = max(pos.stop_loss, pos.trail_stop)
            if   hd >= pos.target:    to_close.append((ticker, pos.target,    "target_hit"))
            elif ld <= pos.stop_loss: to_close.append((ticker, pos.stop_loss, "stop_loss"))
            elif days_held >= MAX_HOLD_DAYS: to_close.append((ticker, cd,     "max_hold"))

        for ticker, exit_p, reason in to_close:
            pos = positions.pop(ticker)
            pnl = (exit_p - pos.entry_price) * pos.shares
            capital += exit_p * pos.shares
            closed.append(Trade(ticker=ticker, entry_date=pos.entry_date,
                exit_date=date, entry_price=pos.entry_price, exit_price=exit_p,
                shares=pos.shares, pnl=pnl,
                pnl_pct=(exit_p - pos.entry_price) / pos.entry_price,
                exit_reason=reason, sector=pos.sector))

        if day_idx % 5 == 0:
            bull_regime, _ = get_breadth(price_data, date)

        total_eq = capital + open_value(positions, price_data, date)
        equity_curve.append({"Date": date, "Equity": total_eq,
                              "Cash": capital, "Positions": len(positions),
                              "BullRegime": bull_regime})

        if not bull_regime: continue
        if len(positions) + len(pending) >= MAX_OPEN_POSITIONS: continue
        if month_trades.get(mkey, 0) >= MAX_TRADES_PM: continue

        # ── SIGNAL SCAN ───────────────────────────────────────────────
        candidates = []
        for ticker, df in price_data.items():
            if ticker in positions or ticker in pending: continue
            df_sl = df[df["Date"] <= date].tail(60)
            if len(df_sl) < 30: continue
            ind_sl = {k: v.iloc[-len(df_sl):] for k, v in inds[ticker].items()}
            fired, sig_info = check_signal(ticker, df_sl, ind_sl)
            if not fired: continue
            candidates.append((ticker, sig_info["quality"], sig_info))

        candidates.sort(key=lambda x: x[1], reverse=True)
        slots = MAX_OPEN_POSITIONS - len(positions) - len(pending)
        for ticker, quality, sig_info in candidates[:slots]:
            pending[ticker] = sig_info

    # Close remaining
    last_date = pd.Timestamp(all_dates[-1])
    for ticker, pos in list(positions.items()):
        sl = price_data[ticker][price_data[ticker]["Date"] <= last_date]
        if sl.empty: continue
        exit_p = sl["Close"].iloc[-1]
        pnl    = (exit_p - pos.entry_price) * pos.shares
        capital += exit_p * pos.shares
        closed.append(Trade(ticker=ticker, entry_date=pos.entry_date,
            exit_date=last_date, entry_price=pos.entry_price, exit_price=exit_p,
            shares=pos.shares, pnl=pnl,
            pnl_pct=(exit_p - pos.entry_price) / pos.entry_price,
            exit_reason="end_of_backtest", sector=pos.sector))

    pm = {}
    for t in closed:
        mk = pd.Timestamp(t.entry_date).strftime("%Y-%m")
        pm[mk] = pm.get(mk, 0) + 1
    print(f"  Total trades     : {len(closed)}")
    print(f"  Avg trades/month : {sum(pm.values())/max(len(pm),1):.1f}")
    return closed, pd.DataFrame(equity_curve)


# ── REPORTING ────────────────────────────────────────────────────────────

def print_summary(trades, equity_df):
    if not trades:
        print("  No trades."); return
    td       = pd.DataFrame([vars(t) for t in trades])
    total    = len(td)
    wins     = int((td["pnl"] > 0).sum())
    losses   = int((td["pnl"] <= 0).sum())
    win_rate = wins / total * 100
    final_eq = equity_df["Equity"].iloc[-1]
    max_dd   = ((equity_df["Equity"] - equity_df["Equity"].cummax())
                / equity_df["Equity"].cummax()).min() * 100
    days     = (equity_df["Date"].iloc[-1] - equity_df["Date"].iloc[0]).days
    cagr     = ((final_eq / INITIAL_CAPITAL) ** (365.25 / days) - 1) * 100
    daily_r  = equity_df["Equity"].pct_change().dropna()
    sharpe   = (daily_r.mean() / (daily_r.std() + 1e-8)) * (252 ** 0.5)
    avg_win  = td[td["pnl"]>0]["pnl_pct"].mean()*100 if wins>0 else 0
    avg_loss = td[td["pnl"]<=0]["pnl_pct"].mean()*100 if losses>0 else 0
    expect   = (win_rate/100)*avg_win + (1-win_rate/100)*avg_loss
    rr       = abs(avg_win/(avg_loss+1e-8))
    td["month"] = pd.to_datetime(td["entry_date"]).dt.to_period("M")
    pm_count = td.groupby("month").size()
    sep = "="*62; sep2 = "-"*62
    lines = ["", sep,
        "  ZODIC  —  EMA Cross + RSI  |  Next-Day Open Entry  (v8)", sep,
        "  Signal: EMA9>EMA21 + RSI 50-70 + ADX>18 + EMA50",
        "  Entry : next-day OPEN  |  Stop: below signal candle low",
        sep2,
        f"  Period       : {equity_df['Date'].iloc[0].date()}  to  {equity_df['Date'].iloc[-1].date()}",
        f"  Initial cap  : INR {INITIAL_CAPITAL:,}",
        f"  Final equity : INR {final_eq:,.0f}",
        f"  CAGR         : {cagr:.2f}%",
        f"  Max Drawdown : {max_dd:.2f}%",
        f"  Sharpe Ratio : {sharpe:.3f}",
        sep2,
        f"  Total trades : {total}",
        f"  Avg/month    : {pm_count.mean():.1f}  (max {pm_count.max()}, min {pm_count.min()})",
        f"  Win rate     : {win_rate:.1f}%  ({wins}W / {losses}L)",
        f"  Avg win      : {avg_win:+.2f}%",
        f"  Avg loss     : {avg_loss:+.2f}%",
        f"  Expectancy   : {expect:+.3f}% per trade",
        f"  R:R ratio    : {rr:.2f}:1",
        sep2,
        "  Exit breakdown:",
        td["exit_reason"].value_counts().to_string(),
        sep2,
        "  Monthly trade counts:",
        pm_count.to_string(), sep]
    summary = "\n".join(str(x) for x in lines)
    print(summary)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    td.to_csv(REPORT_CSV, index=False)
    equity_df.to_csv(EQUITY_CSV, index=False)
    with open(SUMMARY_TXT, "w") as f: f.write(summary)
    print(f"\n  Trade log  -> {REPORT_CSV}")
    print(f"  Equity CSV -> {EQUITY_CSV}")
    print(f"  Summary    -> {SUMMARY_TXT}")


if __name__ == "__main__":
    print("="*62)
    print("  ZODIC  EMA Cross + RSI  Next-Day Open Entry  (v8)")
    print("="*62)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n[1/3] Loading price data...")
    price_data = load_all_prices()
    print("[2/3] Pre-computing indicators...")
    indicators = precompute_indicators(price_data)
    print("[3/3] Running backtest...")
    trades, equity_df = run_backtest(price_data, indicators)
    print_summary(trades, equity_df)
