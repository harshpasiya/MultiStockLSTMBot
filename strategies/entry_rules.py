import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SetupSignal:
    ticker      : str
    date        : pd.Timestamp
    setup_type  : str        # 'breakout' | 'pullback' | 'macd_cross'
    strength    : float      # 0.0 - 1.0  composite quality score
    entry_price : float
    stop_loss   : float      # ATR-based hard stop
    notes       : str        # human-readable reason


class EntryRules:
    """
    3 institutional-grade TA entry rules for NSE 5-day swing trading.

    Rule 1 — MOMENTUM BREAKOUT
      Close > 20-day high  +  Volume > 1.5x avg  +  above 50-MA
      Captures institutional accumulation breakouts.

    Rule 2 — PULLBACK TO MA  (Trend Continuation)
      Price in uptrend (above 50-MA) pulled back to 20-MA  +  RSI 38-55
      Best risk/reward entry — buying dips in uptrend.

    Rule 3 — MACD CROSS + ADX CONFIRM
      MACD histogram flips neg→pos today  +  ADX > 20  +  above 20-MA
      Catches early momentum shifts before full price move.
    """

    def __init__(
        self,
        breakout_lookback : int   = 20,
        vol_multiplier    : float = 1.5,
        ma_proximity_pct  : float = 0.02,
        rsi_low           : float = 38.0,
        rsi_high          : float = 55.0,
        adx_min           : float = 20.0,
        atr_stop_mult     : float = 1.5,
        min_strength      : float = 0.30,
    ):
        self.breakout_lookback = breakout_lookback
        self.vol_multiplier    = vol_multiplier
        self.ma_proximity_pct  = ma_proximity_pct
        self.rsi_low           = rsi_low
        self.rsi_high          = rsi_high
        self.adx_min           = adx_min
        self.atr_stop_mult     = atr_stop_mult
        self.min_strength      = min_strength

    # ── INDICATOR HELPERS ─────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-8)))

    @staticmethod
    def _atr(high, low, close, period: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=10).mean()

    @staticmethod
    def _adx(high, low, close, period: int = 14) -> pd.Series:
        tr       = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        up, down = high - high.shift(1), low.shift(1) - low
        pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
        mdm  = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
        atr  = tr.ewm(alpha=1/period, min_periods=period).mean()
        pdi  = 100 * pdm.ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-8)
        mdi  = 100 * mdm.ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-8)
        dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-8)
        return dx.ewm(alpha=1/period, min_periods=period).mean()

    @staticmethod
    def _macd_hist(close, atr, fast=12, slow=26, signal=9) -> pd.Series:
        ema_f = close.ewm(span=fast,   min_periods=fast).mean()
        ema_s = close.ewm(span=slow,   min_periods=slow).mean()
        macd  = ema_f - ema_s
        sig   = macd.ewm(span=signal,  min_periods=signal).mean()
        return (macd - sig) / (atr + 1e-8)

    # ── RULE 1: MOMENTUM BREAKOUT ─────────────────────────────────────────────

    def rule_breakout(self, df: pd.DataFrame, ticker: str) -> Optional[SetupSignal]:
        if len(df) < 60:
            return None
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

        high_20    = c.shift(1).rolling(self.breakout_lookback).max().iloc[-1]
        vol_avg    = v.rolling(20).mean().iloc[-1]
        ma50       = c.rolling(50).mean().iloc[-1]
        atr        = self._atr(h, l, c).iloc[-1]
        adx        = self._adx(h, l, c).iloc[-1]
        close_now  = c.iloc[-1]
        vol_now    = v.iloc[-1]

        if any(pd.isna(x) for x in [high_20, vol_avg, ma50, atr]):
            return None
        if close_now <= high_20:            return None
        if vol_now < self.vol_multiplier * vol_avg:  return None
        if close_now < ma50:                return None

        brk_mag    = (close_now - high_20) / (high_20 + 1e-8)
        vol_score  = min(vol_now / (vol_avg + 1e-8), 5.0) / 5.0
        adx_score  = min(adx / 40.0, 1.0)
        strength   = (0.40 * min(brk_mag / 0.03, 1.0) +
                      0.35 * vol_score +
                      0.25 * adx_score)

        if strength < self.min_strength:
            return None

        return SetupSignal(
            ticker=ticker, date=df["Date"].iloc[-1],
            setup_type="breakout", strength=round(strength, 4),
            entry_price=close_now,
            stop_loss=round(close_now - self.atr_stop_mult * atr, 2),
            notes=(f"Breakout {brk_mag*100:.2f}% above 20d-high | "
                   f"Vol {vol_now/vol_avg:.1f}x avg | ADX {adx:.1f}"),
        )

    # ── RULE 2: PULLBACK TO MA ────────────────────────────────────────────────

    def rule_pullback(self, df: pd.DataFrame, ticker: str) -> Optional[SetupSignal]:
        if len(df) < 60:
            return None
        c, h, l = df["Close"], df["High"], df["Low"]

        ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
        rsi        = self._rsi(c)
        atr        = self._atr(h, l, c)

        ma20_now   = ma20.iloc[-1]
        ma50_now   = ma50.iloc[-1]
        rsi_now    = rsi.iloc[-1]
        atr_now    = atr.iloc[-1]
        close_now  = c.iloc[-1]
        ma50_slope = (ma50.iloc[-1] - ma50.iloc[-5]) / (ma50.iloc[-5] + 1e-8)

        if any(pd.isna(x) for x in [ma20_now, ma50_now, rsi_now, atr_now]):
            return None
        if close_now < ma50_now:            return None
        dist = (close_now - ma20_now) / (ma20_now + 1e-8)
        if not (-0.005 <= dist <= self.ma_proximity_pct):  return None
        if not (self.rsi_low <= rsi_now <= self.rsi_high): return None
        if ma50_slope < 0:                  return None

        prox_score  = 1.0 - abs(dist) / self.ma_proximity_pct
        rsi_score   = 1.0 - (rsi_now - self.rsi_low) / (self.rsi_high - self.rsi_low)
        slope_score = min(ma50_slope / 0.005, 1.0)
        strength    = (0.40 * prox_score +
                       0.35 * rsi_score +
                       0.25 * slope_score)

        if strength < self.min_strength:
            return None

        return SetupSignal(
            ticker=ticker, date=df["Date"].iloc[-1],
            setup_type="pullback", strength=round(strength, 4),
            entry_price=close_now,
            stop_loss=round(ma50_now - atr_now * 0.5, 2),
            notes=(f"Pullback dist={dist*100:.2f}% | "
                   f"RSI={rsi_now:.1f} | MA50 slope={ma50_slope*100:.3f}%/day"),
        )

    # ── RULE 3: MACD CROSS + ADX ─────────────────────────────────────────────

    def rule_macd_cross(self, df: pd.DataFrame, ticker: str) -> Optional[SetupSignal]:
        if len(df) < 60:
            return None
        c, h, l = df["Close"], df["High"], df["Low"]

        atr       = self._atr(h, l, c)
        macd_h    = self._macd_hist(c, atr)
        adx       = self._adx(h, l, c)
        ma20      = c.rolling(20).mean()

        macd_now  = macd_h.iloc[-1]
        macd_prev = macd_h.iloc[-2]
        adx_now   = adx.iloc[-1]
        atr_now   = atr.iloc[-1]
        close_now = c.iloc[-1]
        ma20_now  = ma20.iloc[-1]

        if any(pd.isna(x) for x in [macd_now, macd_prev, adx_now, ma20_now]):
            return None
        if not (macd_prev < 0 < macd_now):  return None   # cross must happen TODAY
        if adx_now < self.adx_min:          return None
        if close_now < ma20_now:            return None

        cross_mag   = abs(macd_now - macd_prev)
        adx_score   = min((adx_now - self.adx_min) / 20.0, 1.0)
        cross_score = min(cross_mag / 0.002, 1.0)
        price_score = min((close_now - ma20_now) / (ma20_now + 1e-8) / 0.05, 1.0)
        strength    = (0.40 * adx_score +
                       0.35 * cross_score +
                       0.25 * price_score)

        if strength < self.min_strength:
            return None

        return SetupSignal(
            ticker=ticker, date=df["Date"].iloc[-1],
            setup_type="macd_cross", strength=round(strength, 4),
            entry_price=close_now,
            stop_loss=round(close_now - self.atr_stop_mult * atr_now, 2),
            notes=(f"MACD cross neg→pos | ADX={adx_now:.1f} | "
                   f"cross_mag={cross_mag:.5f}"),
        )

    # ── SCAN ALL RULES ────────────────────────────────────────────────────────

    def scan(self, df: pd.DataFrame, ticker: str) -> Optional[SetupSignal]:
        """Run all 3 rules. Return highest-strength signal or None."""
        signals = []
        for rule_fn in [self.rule_breakout, self.rule_pullback, self.rule_macd_cross]:
            try:
                sig = rule_fn(df, ticker)
                if sig is not None:
                    signals.append(sig)
            except Exception:
                continue
        return max(signals, key=lambda s: s.strength) if signals else None


# ── BATCH SCANNER (used by backtest + live) ───────────────────────────────────

def scan_universe(price_data: dict, date: pd.Timestamp,
                  lookback_days: int = 120) -> list:
    """
    Scan all tickers for a given date.
    price_data[ticker] = DataFrame [Date, Open, High, Low, Close, Volume]
    Returns list of SetupSignal sorted by strength descending.
    """
    rules, signals = EntryRules(), []
    for ticker, df in price_data.items():
        df_sl = df[df["Date"] <= date].tail(lookback_days).copy()
        if len(df_sl) < 60:
            continue
        sig = rules.scan(df_sl, ticker)
        if sig is not None:
            signals.append(sig)
    return sorted(signals, key=lambda s: s.strength, reverse=True)
