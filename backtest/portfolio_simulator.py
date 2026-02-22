import os, sys, glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.lstm_ranker import ZodicLSTMRanker

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = ROOT / "data" / "grouped_sequences.npy"
RAW_DIR     = ROOT / "data"
MODEL_PATH  = ROOT / "models" / "zodic_omega_ranker.pt"
EQUITY_CSV  = ROOT / "outputs" / "equity_curve.csv"
TRADES_CSV  = ROOT / "outputs" / "trade_log.csv"

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
INITIAL_CAP    = 100_000.0

CORE_SLOTS     = 2
FAST_SLOTS     = 2
TOTAL_SLOTS    = CORE_SLOTS + FAST_SLOTS

HOLD_DAYS_CORE = 20
HOLD_DAYS_FAST = 7

STOP_LOSS_CORE = -0.04       # 4% hard stop (was None)
STOP_LOSS_FAST = -0.03       # 3% hard stop (was -2%, too tight)

TRAIL_TRIGGER_CORE = 0.04
TRAIL_PCT_CORE     = 0.12
TRAIL_TRIGGER_FAST = 0.025
TRAIL_PCT_FAST     = 0.06

COST_BPS       = 10
TOP_PCT        = 0.25        # wider funnel — filter handles quality

# Technical entry filter thresholds
RSI_MIN        = 40
RSI_MAX        = 70
MA_PERIOD      = 50
BREADTH_MIN    = 0.50
INDEX_MOM_MIN  = 0.0

TEST_FRAC      = 0.15
DD_HIGH_RISK   = 0.12


# ── TECHNICAL ENTRY FILTER ────────────────────────────────────────────────────

class TechnicalEntryFilter:
    """
    Blocks entry into declining or extended stocks.
    Computed once at init from loaded price data — zero extra I/O.

    Conditions checked on entry date:
      1. Close > 50-day MA       → stock is in uptrend
      2. RSI 14 between 40–70   → not oversold or overbought
      3. 3-day return > -5%     → no sharp recent breakdown
    """

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.ma50   = prices.rolling(MA_PERIOD, min_periods=30).mean()
        self.rsi    = self._rsi(prices, 14)
        self.ret3   = prices.pct_change(3)

    @staticmethod
    def _rsi(prices: pd.DataFrame, period: int) -> pd.DataFrame:
        delta = prices.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-8)))

    def passes(self, ticker: str, date) -> bool:
        try:
            if ticker not in self.prices.columns or date not in self.prices.index:
                return True
            close = self.prices.loc[date, ticker]
            ma50  = self.ma50.loc[date, ticker]
            rsi   = self.rsi.loc[date, ticker]
            r3    = self.ret3.loc[date, ticker]
            if any(pd.isna(x) for x in [close, ma50, rsi]):
                return True
            return (close > ma50) and (RSI_MIN <= rsi <= RSI_MAX) and (pd.isna(r3) or r3 > -0.05)
        except Exception:
            return True


# ── REGIME FILTER ─────────────────────────────────────────────────────────────

def build_regime(prices: pd.DataFrame) -> pd.DataFrame:
    idx     = prices.mean(axis=1)
    ret20   = idx.pct_change(20)
    breadth = (prices.pct_change(20) > 0).mean(axis=1)
    return pd.DataFrame({"ret20": ret20, "breadth": breadth}).dropna()


def regime_bullish(regime: pd.DataFrame, date) -> bool:
    if date not in regime.index:
        return False
    r = regime.loc[date]
    return bool(r["breadth"] >= BREADTH_MIN and r["ret20"] > INDEX_MOM_MIN)


# ── PRICE LOADER ──────────────────────────────────────────────────────────────

def load_actual_prices():
    frames = {}
    for path in glob.glob(str(RAW_DIR / "*_NS_raw.csv")):
        ticker = os.path.basename(path).replace("_NS_raw.csv", "")
        try:
            df = pd.read_csv(path, skiprows=2)
            df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
            df["Date"]  = pd.to_datetime(df["Date"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Close"]).sort_values("Date")
            frames[ticker] = df.set_index("Date")["Close"]
        except Exception:
            continue
    prices = pd.DataFrame(frames).sort_index()
    print(f"Loaded prices : {prices.shape[0]} dates x {prices.shape[1]} tickers")
    return prices


# ── POSITION ──────────────────────────────────────────────────────────────────

class Position:
    def __init__(self, ticker, entry_price, entry_date, kind):
        self.ticker      = ticker
        self.entry_price = entry_price
        self.entry_date  = entry_date
        self.kind        = kind
        self.max_price   = entry_price
        self.prev_price  = entry_price
        self.days_held   = 0

    def step(self, px):
        self.days_held += 1
        self.max_price  = max(self.max_price, px)
        daily_ret       = px / self.prev_price - 1
        total_ret       = px / self.entry_price - 1
        self.prev_price = px

        stop  = STOP_LOSS_CORE  if self.kind == "core" else STOP_LOSS_FAST
        trig  = TRAIL_TRIGGER_CORE if self.kind == "core" else TRAIL_TRIGGER_FAST
        trail = TRAIL_PCT_CORE  if self.kind == "core" else TRAIL_PCT_FAST
        maxh  = HOLD_DAYS_CORE  if self.kind == "core" else HOLD_DAYS_FAST

        if total_ret <= stop:
            return True, daily_ret, total_ret
        if self.max_price / self.entry_price >= (1 + trig):
            if px / self.max_price - 1 <= -trail:
                return True, daily_ret, total_ret
        if self.days_held >= maxh:
            return True, daily_ret, total_ret
        return False, daily_ret, total_ret

    def exit_reason(self, px):
        stop  = STOP_LOSS_CORE  if self.kind == "core" else STOP_LOSS_FAST
        trig  = TRAIL_TRIGGER_CORE if self.kind == "core" else TRAIL_TRIGGER_FAST
        trail = TRAIL_PCT_CORE  if self.kind == "core" else TRAIL_PCT_FAST
        t     = px / self.entry_price - 1
        if t <= stop:
            return "STOP_LOSS"
        if self.max_price / self.entry_price >= (1 + trig):
            if px / self.max_price - 1 <= -trail:
                return "TRAIL_STOP"
        return "TIME_EXIT"


# ── BACKTEST LOOP ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run():
    print(f"Device : {DEVICE}")
    grouped    = np.load(DATA_PATH, allow_pickle=True).item()
    all_dates  = sorted(grouped.keys())
    n_features = grouped[all_dates[0]][0]["seq"].shape[-1]

    model = ZodicLSTMRanker(n_features=n_features).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded  : {n_features} features")

    prices = load_actual_prices()
    tech   = TechnicalEntryFilter(prices)
    regime = build_regime(prices)

    n          = len(all_dates)
    test_dates = all_dates[int((1 - TEST_FRAC) * n):]
    bull_days  = sum(1 for d in test_dates if regime_bullish(regime, d))
    print(f"Backtest      : {test_dates[0].date()} to {test_dates[-1].date()}")
    print(f"Test days     : {len(test_dates)}")
    print(f"Bullish days  : {bull_days}/{len(test_dates)} "
          f"({100*bull_days/len(test_dates):.0f}%)\n")

    capital   = INITIAL_CAP
    positions = {}
    equity    = []
    trade_log = []
    eq_hist   = []

    for date in test_dates:
        if date not in prices.index:
            equity.append(capital);  eq_hist.append(capital);  continue

        day_px = prices.loc[date]

        # Dynamic risk sizing
        if len(eq_hist) > 60:
            es = pd.Series(eq_hist)
            dd = float((es / es.cummax() - 1).min())
            core_s = max(1, CORE_SLOTS - (1 if dd < -DD_HIGH_RISK else 0))
            fast_s = max(1, FAST_SLOTS - (1 if dd < -DD_HIGH_RISK else 0))
        else:
            core_s, fast_s = CORE_SLOTS, FAST_SLOTS

        pos_weight = 1.0 / (core_s + fast_s)

        # Mark-to-market + exits
        daily_pnl = 0.0
        exits = []
        for ticker, pos in list(positions.items()):
            px = day_px.get(ticker)
            if px is None or pd.isna(px) or px <= 0:
                continue
            exit_flag, d_ret, t_ret = pos.step(float(px))
            daily_pnl += pos_weight * d_ret
            if exit_flag:
                trade_log.append({
                    "Ticker":      ticker,
                    "Kind":        pos.kind,
                    "Entry_Date":  str(pos.entry_date.date()),
                    "Exit_Date":   str(date.date()),
                    "Entry_Price": round(pos.entry_price, 2),
                    "Exit_Price":  round(float(px), 2),
                    "Days_Held":   pos.days_held,
                    "Gross_Ret":   round(t_ret, 4),
                    "Net_Ret":     round(t_ret - COST_BPS/10000, 4),
                    "Exit_Reason": pos.exit_reason(float(px)),
                })
                exits.append(ticker)
        for t in exits:
            positions.pop(t, None)
        capital *= (1 + daily_pnl)

        # Regime gate
        if not regime_bullish(regime, date):
            equity.append(capital);  eq_hist.append(capital);  continue

        # New entries
        used_core  = sum(1 for p in positions.values() if p.kind == "core")
        used_fast  = sum(1 for p in positions.values() if p.kind == "fast")
        avail_core = core_s - used_core
        avail_fast = fast_s - used_fast

        if (avail_core > 0 or avail_fast > 0) and date in grouped:
            day_data = grouped[date]
            if len(day_data) >= 5:
                seqs    = torch.tensor(
                    np.array([s["seq"] for s in day_data]),
                    dtype=torch.float32).to(DEVICE)
                # Score all stocks with probability
                logits = model(seqs)
                probs = torch.sigmoid(logits).cpu().numpy()
                tickers = [s["ticker"] for s in day_data]

                # Only consider stocks where model is confident (prob > 0.45)
                ranked = sorted(
                    [(t, float(p)) for t, p in zip(tickers, probs) if float(p) > 0.45],
                    key=lambda x: x[1], reverse=True
                )

                held = set(positions.keys())
                for rank_idx, (ticker, prob) in enumerate(ranked, 1):  # ← no n_top
                    if avail_core <= 0 and avail_fast <= 0:
                        break
                    if ticker in held:
                        continue
                    if not tech.passes(ticker, date):
                        continue
                    px = day_px.get(ticker)
                    if px is None or pd.isna(px) or float(px) <= 0:
                        continue
                    kind = "core" if (rank_idx <= core_s and avail_core > 0) else "fast"
                    if kind == "fast" and avail_fast <= 0:
                        continue
                    if kind == "core":
                        avail_core -= 1
                    else:
                        avail_fast -= 1
                    capital -= capital * pos_weight * (COST_BPS / 10000)
                    positions[ticker] = Position(ticker, float(px), date, kind)
                    held.add(ticker)

    # ── METRICS ───────────────────────────────────────────────────────────────
    eq        = pd.Series(equity, index=test_dates[:len(equity)])
    returns   = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / INITIAL_CAP) - 1
    n_years   = len(eq) / 252.0
    cagr      = ((eq.iloc[-1] / INITIAL_CAP) ** (1.0 / max(n_years, 1e-9))) - 1
    sharpe    = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
    max_dd    = (eq / eq.cummax() - 1).min()
    tdf       = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    n_trades  = len(tdf)
    win_rate  = float((tdf["Net_Ret"] > 0).mean()) if n_trades else 0
    avg_win   = float(tdf.loc[tdf["Net_Ret"] > 0,  "Net_Ret"].mean()) if n_trades else 0
    avg_loss  = float(tdf.loc[tdf["Net_Ret"] <= 0, "Net_Ret"].mean()) if n_trades else 0
    pf        = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    print("\n" + "=" * 52)
    print(" BACKTEST RESULTS")
    print("=" * 52)
    print(f" Period          : {test_dates[0].date()} to {test_dates[-1].date()}")
    print(f" Initial Capital : Rs {INITIAL_CAP:,.0f}")
    print(f" Final Capital   : Rs {eq.iloc[-1]:,.0f}")
    print(f" Total Return    : {total_ret*100:.2f}%")
    print(f" CAGR            : {cagr*100:.2f}%  [target 45-50%]")
    print(f" Sharpe Ratio    : {sharpe:.3f}")
    print(f" Max Drawdown    : {max_dd*100:.2f}%")
    print(f" Trades          : {n_trades}")
    print(f" Win Rate        : {win_rate*100:.1f}%")
    print(f" Avg Win         : {avg_win*100:.2f}%")
    print(f" Avg Loss        : {avg_loss*100:.2f}%")
    print(f" Profit Factor   : {pf:.2f}")
    if n_trades:
        by_r = tdf["Exit_Reason"].value_counts()
        by_k = tdf.get("Kind", pd.Series()).value_counts()
        print(f" Stop Loss hits  : {by_r.get('STOP_LOSS', 0)}")
        print(f" Trail Stop hits : {by_r.get('TRAIL_STOP', 0)}")
        print(f" Time Exits      : {by_r.get('TIME_EXIT', 0)}")
        print(f" Core trades     : {by_k.get('core', 0)}")
        print(f" Fast trades     : {by_k.get('fast', 0)}")
    print("=" * 52)

    os.makedirs(str(EQUITY_CSV.parent), exist_ok=True)
    eq.to_csv(EQUITY_CSV)
    if n_trades:
        tdf.to_csv(TRADES_CSV, index=False)
    print(f"\nEquity curve -> {EQUITY_CSV}")
    print(f"Trade log    -> {TRADES_CSV}")
    return eq, tdf


if __name__ == "__main__":
    run()
