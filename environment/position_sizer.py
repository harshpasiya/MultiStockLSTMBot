"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Adaptive Position Sizer                         ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : environment/position_sizer.py                          ║
║         Phase   : 3 — RL Agent Training / Live Trading                  ║
║                                                                          ║
║  What this module does:                                                  ║
║    Computes the optimal position size for each trade using the           ║
║    Fractional Kelly Criterion, adjusted by:                              ║
║      • RL agent confidence score (higher confidence → larger size)       ║
║      • Current volatility regime (high vol → smaller size)              ║
║      • Portfolio heat (approaching max heat → shrink new positions)      ║
║      • MDS market direction score (bearish market → shrink longs)        ║
║      • Drawdown regime (in drawdown → defensive sizing)                  ║
║                                                                          ║
║  Why Kelly Criterion?                                                    ║
║    Fixed position sizing (always 25%) leaves significant CAGR on the    ║
║    table. Kelly optimally allocates more capital to higher-confidence    ║
║    trades and less to borderline ones. Fractional Kelly (quarter-Kelly)  ║
║    provides ~75% of the CAGR benefit with dramatically lower variance.  ║
║                                                                          ║
║  Kelly Formula:                                                          ║
║    f* = (b × p − q) / b                                                 ║
║    where:                                                                ║
║      b = reward-to-risk ratio (TP% / SL%)                               ║
║      p = estimated win probability (from RL confidence score)            ║
║      q = 1 - p (loss probability)                                        ║
║                                                                          ║
║    Quarter Kelly: position_size = 0.25 × f*                             ║
║                                                                          ║
║  Output:                                                                 ║
║    SizingResult with:                                                    ║
║      position_pct   : fraction of portfolio to invest [0, MAX_PCT]      ║
║      invest_amount  : ₹ amount to invest                                 ║
║      quantity       : number of shares to buy                            ║
║      kelly_fraction : raw Kelly fraction (before adjustments)            ║
║      adjustments    : dict of all scaling factors applied                ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install numpy loguru                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from loguru import logger


# ── Position sizing limits ────────────────────────────────────────────────
MIN_POSITION_PCT = 0.05    # never invest less than 5% of portfolio per trade
MAX_POSITION_PCT = 0.25    # never invest more than 25% of portfolio per trade
MAX_PORTFOLIO_HEAT = 0.06  # total open risk cap (matches Risk Constitution)
MIN_INVEST_INR   = 5_000   # minimum ₹5,000 per trade (practical floor)

# ── Kelly fractions by confidence tier ───────────────────────────────────
# Quarter-Kelly = 0.25 is the base. Scales with confidence.
KELLY_FRACTION_BASE = 0.25

# ── MDS position size multipliers (from roadmap spec) ────────────────────
MDS_MULTIPLIERS = {
    3 : 1.5,   # Strong bullish: 1.5× size
    2 : 1.2,   # Bullish: 1.2×
    1 : 1.0,   # Mild bullish: normal
    0 : 1.0,   # Neutral: normal
   -1 : 0.8,   # Mild bearish: 0.8×
   -2 : 0.5,   # Bearish: 0.5×
   -3 : 0.0,   # Strong bearish: no new longs
}

# ── Volatility regime multipliers ────────────────────────────────────────
# When ATR% is high, reduce size to keep ₹ risk per trade roughly constant
VOL_REGIME_MULTIPLIERS = {
    "low"    : 1.2,   # Low vol: slightly larger size
    "normal" : 1.0,   # Normal: standard size
    "high"   : 0.7,   # High vol: reduce to keep risk constant
    "extreme": 0.4,   # Extreme vol (>4% ATR): very small
}

# ── Drawdown regime multipliers ───────────────────────────────────────────
DRAWDOWN_MULTIPLIERS = {
    (0.00, 0.05): 1.0,    # No significant drawdown: normal
    (0.05, 0.08): 0.75,   # Caution: reduce to 75%
    (0.08, 0.10): 0.50,   # Warning: reduce to 50%
    (0.10, 0.12): 0.25,   # Critical: minimum sizing only
}


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SizingInput:
    """
    All inputs required to compute position size for one trade.

    Populated by the signal engine before calling PositionSizer.compute().
    """
    # ── Trade parameters ──────────────────────────────────────────────────
    symbol          : str
    entry_price     : float          # ₹ entry price
    tp_price        : float          # ₹ take-profit price
    sl_price        : float          # ₹ stop-loss price

    # ── RL agent output ───────────────────────────────────────────────────
    confidence_score: float          # RL agent confidence [0.0, 1.0]
                                     # 0.55 = borderline, 0.85 = very high

    # ── Portfolio state ───────────────────────────────────────────────────
    portfolio_value : float          # current total portfolio value ₹
    available_cash  : float          # cash available for new trades ₹
    current_drawdown: float          # current drawdown from peak [0, 1]
    open_risk_pct   : float          # current portfolio heat [0, 1]
                                     # = sum of (SL% × position_size%) for all open

    # ── Market context ────────────────────────────────────────────────────
    mds_score       : int            # Market Direction Signal [-3, +3]
    vol_regime      : str = "normal" # 'low', 'normal', 'high', 'extreme'
    atr_pct         : float = 0.02   # ATR as % of price (e.g. 0.02 = 2%)

    # ── Trade mode ────────────────────────────────────────────────────────
    is_strong_signal: bool  = False  # Strong Buy vs Buy (affects Kelly fraction)


@dataclass
class SizingResult:
    """
    Output of position sizing computation.

    All monetary values in ₹. position_pct is fraction of portfolio.
    """
    symbol          : str
    position_pct    : float          # fraction of portfolio [0, MAX_POSITION_PCT]
    invest_amount   : float          # ₹ to invest
    quantity        : int            # shares to buy (floor division)
    actual_pct      : float          # actual % after quantity rounding

    # ── Sizing components (for transparency/logging) ──────────────────────
    kelly_fraction  : float          # raw f* from Kelly formula
    quarter_kelly   : float          # 0.25 × f* (before adjustments)
    win_probability : float          # estimated p from confidence score
    reward_risk_ratio: float         # b = TP% / SL%

    # ── Adjustment factors applied ────────────────────────────────────────
    adjustments     : Dict[str, float] = field(default_factory=dict)

    # ── Flags ─────────────────────────────────────────────────────────────
    is_minimum_size : bool  = False  # True if clamped to minimum
    is_zero         : bool  = False  # True if sizing returned 0 (don't trade)
    reason          : str   = ""     # explanation if is_zero=True

    @property
    def risk_amount(self) -> float:
        """₹ at risk on this trade (invest_amount × SL%)."""
        return self.invest_amount * (1 - self.position_pct)  # approximate

    def summary(self) -> str:
        adj_str = " × ".join(
            f"{k}={v:.2f}" for k, v in self.adjustments.items() if v != 1.0
        )
        return (
            f"{self.symbol}: {self.position_pct:.1%} of portfolio "
            f"| ₹{self.invest_amount:,.0f} × {self.quantity} shares "
            f"| Kelly={self.kelly_fraction:.3f} → Q-Kelly={self.quarter_kelly:.3f}"
            + (f" | adj: {adj_str}" if adj_str else "")
            + (f" | ⚠ {self.reason}" if self.is_zero else "")
        )


# ══════════════════════════════════════════════════════════════════════════
#  POSITION SIZER
# ══════════════════════════════════════════════════════════════════════════

class PositionSizer:
    """
    Computes fractional Kelly position sizes with multi-factor adjustments.

    This is the module that converts a trading signal into a specific
    rupee amount and share quantity. It is called by the signal engine
    for every new trade entry.

    Design principles:
        1. Start with full Kelly fraction
        2. Apply quarter-Kelly base (conservative, proven effective)
        3. Scale by confidence (higher confidence → closer to full quarter-Kelly)
        4. Apply MDS market direction multiplier
        5. Apply volatility regime multiplier
        6. Apply drawdown protection multiplier
        7. Clip to [MIN_POSITION_PCT, MAX_POSITION_PCT]
        8. Further reduce if would breach portfolio heat limit
        9. Convert to ₹ amount and share quantity

    Usage:
        sizer = PositionSizer()

        result = sizer.compute(SizingInput(
            symbol           = "RELIANCE",
            entry_price      = 2850.0,
            tp_price         = 2964.0,    # 4% TP
            sl_price         = 2807.25,   # 1.5% SL
            confidence_score = 0.72,
            portfolio_value  = 1_000_000,
            available_cash   = 800_000,
            current_drawdown = 0.02,
            open_risk_pct    = 0.02,
            mds_score        = 1,
            vol_regime       = "normal",
            atr_pct          = 0.018,
        ))

        if result.is_zero:
            logger.info(f"Skip trade: {result.reason}")
        else:
            place_order(result.symbol, result.quantity, result.invest_amount)
    """

    def __init__(
        self,
        kelly_fraction : float = KELLY_FRACTION_BASE,
        min_pct        : float = MIN_POSITION_PCT,
        max_pct        : float = MAX_POSITION_PCT,
        max_heat       : float = MAX_PORTFOLIO_HEAT,
        min_invest_inr : float = MIN_INVEST_INR,
    ):
        self.kelly_fraction  = kelly_fraction
        self.min_pct         = min_pct
        self.max_pct         = max_pct
        self.max_heat        = max_heat
        self.min_invest_inr  = min_invest_inr

        # Running history for win rate tracking
        self._trade_outcomes : list[bool]  = []   # True = win, False = loss
        self._confidence_bins: Dict[str, list[bool]] = {
            "high"  : [],   # confidence > 0.75
            "medium": [],   # confidence 0.60 – 0.75
            "low"   : [],   # confidence 0.55 – 0.60
        }

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN COMPUTE METHOD
    # ══════════════════════════════════════════════════════════════════════

    def compute(self, inp: SizingInput) -> SizingResult:
        """
        Computes the optimal fractional Kelly position size.

        Args:
            inp : SizingInput with all required trade and market parameters

        Returns:
            SizingResult with position_pct, invest_amount, quantity, and
            all intermediate calculations for transparency.
        """
        adjustments: Dict[str, float] = {}

        # ── Step 1: Validate inputs ───────────────────────────────────────
        validation = self._validate(inp)
        if validation is not None:
            return validation   # returns is_zero=True result

        # ── Step 2: Compute raw Kelly fraction ────────────────────────────
        reward_risk_ratio, kelly_raw, win_prob = self._compute_kelly(inp)

        # ── Step 3: Apply quarter-Kelly base ──────────────────────────────
        quarter_k = kelly_raw * self.kelly_fraction

        # ── Step 4: Scale by confidence score ────────────────────────────
        # Maps confidence [0.55, 1.0] → scale [0.5, 1.0]
        # Below 0.55 = no trade (filtered before reaching here)
        conf_scale = self._confidence_scale(inp.confidence_score)
        adjustments["confidence"] = conf_scale

        # Strong signal gets a 1.2× boost
        if inp.is_strong_signal:
            adjustments["strong_signal"] = 1.2
        else:
            adjustments["strong_signal"] = 1.0

        # ── Step 5: MDS market direction multiplier ───────────────────────
        mds_mult = MDS_MULTIPLIERS.get(inp.mds_score, 1.0)
        if mds_mult == 0.0:
            return self._zero_result(
                inp, kelly_raw, quarter_k, win_prob, reward_risk_ratio,
                adjustments, reason=f"MDS score {inp.mds_score} = no new longs"
            )
        adjustments["mds"] = mds_mult

        # ── Step 6: Volatility regime multiplier ──────────────────────────
        vol_mult = VOL_REGIME_MULTIPLIERS.get(inp.vol_regime, 1.0)

        # Additional ATR-based scaling: if ATR% > 3%, reduce further
        if inp.atr_pct > 0.04:
            vol_mult *= 0.6
            adjustments["atr_extreme"] = 0.6
        elif inp.atr_pct > 0.03:
            vol_mult *= 0.8
            adjustments["atr_high"] = 0.8

        adjustments["volatility"] = vol_mult

        # ── Step 7: Drawdown protection multiplier ────────────────────────
        dd_mult = self._drawdown_multiplier(inp.current_drawdown)
        adjustments["drawdown"] = dd_mult

        # ── Step 8: Aggregate all adjustments ────────────────────────────
        total_adj = (
            conf_scale *
            adjustments["strong_signal"] *
            mds_mult *
            vol_mult *
            dd_mult
        )

        raw_position_pct = quarter_k * total_adj

        # ── Step 9: Portfolio heat cap ────────────────────────────────────
        # Don't let this trade push total risk above max_heat
        sl_pct           = (inp.entry_price - inp.sl_price) / inp.entry_price
        new_heat_contrib = raw_position_pct * sl_pct
        remaining_heat   = max(0.0, self.max_heat - inp.open_risk_pct)

        if new_heat_contrib > remaining_heat and sl_pct > 0:
            heat_limited_pct       = remaining_heat / sl_pct
            adjustments["heat_cap"] = heat_limited_pct / max(raw_position_pct, 1e-8)
            raw_position_pct       = heat_limited_pct
        else:
            adjustments["heat_cap"] = 1.0

        # ── Step 10: Clip to [min_pct, max_pct] ──────────────────────────
        is_minimum = False
        if raw_position_pct < self.min_pct and raw_position_pct > 0:
            raw_position_pct = self.min_pct
            is_minimum       = True

        position_pct = float(np.clip(raw_position_pct, 0.0, self.max_pct))

        # ── Step 11: Convert to ₹ amount ─────────────────────────────────
        invest_amount = position_pct * inp.portfolio_value

        # Cannot invest more than available cash (with 2% buffer)
        invest_amount = min(invest_amount, inp.available_cash * 0.98)

        if invest_amount < self.min_invest_inr:
            return self._zero_result(
                inp, kelly_raw, quarter_k, win_prob, reward_risk_ratio,
                adjustments,
                reason=f"Invest amount ₹{invest_amount:.0f} below minimum ₹{self.min_invest_inr:.0f}"
            )

        # ── Step 12: Compute share quantity ──────────────────────────────
        quantity = int(invest_amount // inp.entry_price)

        if quantity < 1:
            return self._zero_result(
                inp, kelly_raw, quarter_k, win_prob, reward_risk_ratio,
                adjustments,
                reason=f"Quantity rounds to 0 at price ₹{inp.entry_price:.2f}"
            )

        actual_invest  = quantity * inp.entry_price
        actual_pct     = actual_invest / inp.portfolio_value

        logger.debug(
            f"PositionSizer: {inp.symbol} "
            f"Kelly={kelly_raw:.3f} → Q-Kelly={quarter_k:.3f} → "
            f"adj={total_adj:.3f} → {actual_pct:.1%} "
            f"| qty={quantity} @ ₹{inp.entry_price:.2f} "
            f"| ₹{actual_invest:,.0f}"
        )

        return SizingResult(
            symbol           = inp.symbol,
            position_pct     = actual_pct,
            invest_amount    = actual_invest,
            quantity         = quantity,
            actual_pct       = actual_pct,
            kelly_fraction   = kelly_raw,
            quarter_kelly    = quarter_k,
            win_probability  = win_prob,
            reward_risk_ratio= reward_risk_ratio,
            adjustments      = adjustments,
            is_minimum_size  = is_minimum,
            is_zero          = False,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  OUTCOME TRACKING (updates win probability estimates over time)
    # ══════════════════════════════════════════════════════════════════════

    def record_outcome(self, confidence_score: float, won: bool):
        """
        Records a completed trade outcome for win-rate tracking.

        Over time this builds a real win-rate estimate per confidence bin,
        which can replace the estimated win_probability in compute().

        Args:
            confidence_score : The confidence score used when the trade was entered
            won              : True if trade hit TP, False if SL or worse
        """
        self._trade_outcomes.append(won)

        if confidence_score >= 0.75:
            self._confidence_bins["high"].append(won)
        elif confidence_score >= 0.60:
            self._confidence_bins["medium"].append(won)
        else:
            self._confidence_bins["low"].append(won)

        # Keep bounded
        for key in self._confidence_bins:
            if len(self._confidence_bins[key]) > 500:
                self._confidence_bins[key] = self._confidence_bins[key][-500:]

    def get_empirical_win_rates(self) -> Dict[str, float]:
        """
        Returns empirical win rates per confidence bin.
        Becomes meaningful after ~50+ trades per bin.
        """
        result = {}
        for bin_name, outcomes in self._confidence_bins.items():
            if len(outcomes) >= 10:
                result[bin_name] = sum(outcomes) / len(outcomes)
            else:
                result[bin_name] = None  # insufficient data
        return result

    def get_overall_stats(self) -> Dict:
        """Returns overall trade statistics."""
        if not self._trade_outcomes:
            return {"n_trades": 0, "win_rate": None, "profit_factor": None}
        n     = len(self._trade_outcomes)
        wins  = sum(self._trade_outcomes)
        return {
            "n_trades" : n,
            "win_rate" : wins / n,
            "n_wins"   : wins,
            "n_losses" : n - wins,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _compute_kelly(
        self,
        inp: SizingInput,
    ) -> Tuple[float, float, float]:
        """
        Computes raw Kelly fraction f* from trade parameters.

        Kelly formula: f* = (b × p − q) / b
            b = reward-to-risk ratio = (TP% / SL%)
            p = win probability (estimated from confidence score)
            q = 1 - p

        Returns:
            (reward_risk_ratio, kelly_fraction, win_probability)
        """
        # Reward-to-risk ratio
        tp_pct = (inp.tp_price - inp.entry_price) / inp.entry_price
        sl_pct = (inp.entry_price - inp.sl_price) / inp.entry_price

        # Guard against zero or negative SL
        sl_pct = max(sl_pct, 0.005)
        tp_pct = max(tp_pct, 0.010)

        b = tp_pct / sl_pct   # reward-to-risk

        # Estimate win probability from confidence score
        # Maps confidence [0.55, 1.0] → win probability [0.55, 0.72]
        # We cap at 0.72 because even the best system rarely exceeds this
        p = self._confidence_to_win_prob(inp.confidence_score)
        q = 1.0 - p

        # Kelly fraction
        kelly = (b * p - q) / b

        # Clip to [0, 1] — negative Kelly means don't trade
        kelly = max(0.0, min(1.0, kelly))

        return b, kelly, p

    def _confidence_to_win_prob(self, confidence: float) -> float:
        """
        Maps RL confidence score [0.55, 1.0] to estimated win probability.

        Based on the Phase 2 pre-training results:
            confidence > 0.85  → estimated win rate ~68%
            confidence 0.70–0.85 → estimated win rate ~63%
            confidence 0.55–0.70 → estimated win rate ~58%

        Uses empirical win rates from trade history when available.
        """
        empirical = self.get_empirical_win_rates()

        if confidence >= 0.75 and empirical.get("high") is not None:
            return empirical["high"]
        elif 0.60 <= confidence < 0.75 and empirical.get("medium") is not None:
            return empirical["medium"]
        elif empirical.get("low") is not None:
            return empirical["low"]

        # Fallback: linear interpolation from confidence to win prob
        # confidence=0.55 → p=0.55, confidence=1.0 → p=0.72
        p = 0.55 + (confidence - 0.55) * (0.72 - 0.55) / (1.0 - 0.55)
        return float(np.clip(p, 0.50, 0.75))

    def _confidence_scale(self, confidence: float) -> float:
        """
        Maps confidence [0.55, 1.0] to a Kelly scaling factor [0.5, 1.0].

        Below 0.55 → trade should not be placed (filtered upstream by RL agent).
        At 0.55 → scale = 0.5 (minimum sizing)
        At 0.85+ → scale = 1.0 (full quarter-Kelly)
        """
        if confidence >= 0.85:
            return 1.0
        elif confidence >= 0.70:
            # Linear: 0.70→0.75, 0.85→1.0
            return 0.75 + (confidence - 0.70) * (1.0 - 0.75) / (0.85 - 0.70)
        elif confidence >= 0.55:
            # Linear: 0.55→0.5, 0.70→0.75
            return 0.50 + (confidence - 0.55) * (0.75 - 0.50) / (0.70 - 0.55)
        else:
            return 0.0   # below threshold — should not reach here

    def _drawdown_multiplier(self, drawdown: float) -> float:
        """
        Returns position size multiplier based on current drawdown level.
        Implements a graduated defense — the deeper the drawdown,
        the smaller the new positions.
        """
        for (low, high), mult in DRAWDOWN_MULTIPLIERS.items():
            if low <= drawdown < high:
                return mult
        if drawdown >= 0.10:
            return 0.25   # critical zone
        return 1.0

    def _validate(self, inp: SizingInput) -> Optional[SizingResult]:
        """
        Validates inputs. Returns a zero SizingResult if invalid,
        None if all inputs are valid (caller should proceed).
        """
        # Confidence too low — no trade
        if inp.confidence_score < 0.55:
            return self._zero_result(
                inp, 0.0, 0.0, 0.0, 0.0, {},
                reason=f"Confidence {inp.confidence_score:.3f} below 0.55 threshold"
            )

        # Price sanity
        if inp.entry_price <= 0:
            return self._zero_result(
                inp, 0.0, 0.0, 0.0, 0.0, {},
                reason="Invalid entry price (≤ 0)"
            )

        # TP must be above entry, SL must be below entry
        if inp.tp_price <= inp.entry_price:
            return self._zero_result(
                inp, 0.0, 0.0, 0.0, 0.0, {},
                reason=f"TP price {inp.tp_price} not above entry {inp.entry_price}"
            )

        if inp.sl_price >= inp.entry_price:
            return self._zero_result(
                inp, 0.0, 0.0, 0.0, 0.0, {},
                reason=f"SL price {inp.sl_price} not below entry {inp.entry_price}"
            )

        # No cash available
        if inp.available_cash < self.min_invest_inr:
            return self._zero_result(
                inp, 0.0, 0.0, 0.0, 0.0, {},
                reason=f"Available cash ₹{inp.available_cash:.0f} below minimum"
            )

        # Already at max portfolio heat
        if inp.open_risk_pct >= self.max_heat:
            return self._zero_result(
                inp, 0.0, 0.0, 0.0, 0.0, {},
                reason=f"Portfolio heat {inp.open_risk_pct:.2%} at maximum {self.max_heat:.2%}"
            )

        return None   # all valid

    def _zero_result(
        self,
        inp              : SizingInput,
        kelly_fraction   : float,
        quarter_kelly    : float,
        win_probability  : float,
        reward_risk_ratio: float,
        adjustments      : Dict,
        reason           : str = "",
    ) -> SizingResult:
        """Factory for a zero-size result (don't trade)."""
        return SizingResult(
            symbol           = inp.symbol,
            position_pct     = 0.0,
            invest_amount    = 0.0,
            quantity         = 0,
            actual_pct       = 0.0,
            kelly_fraction   = kelly_fraction,
            quarter_kelly    = quarter_kelly,
            win_probability  = win_probability,
            reward_risk_ratio= reward_risk_ratio,
            adjustments      = adjustments,
            is_zero          = True,
            reason           = reason,
        )


# ══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FACTORY
# ══════════════════════════════════════════════════════════════════════════

def build_position_sizer(config: Optional[Dict] = None) -> PositionSizer:
    """
    Builds a PositionSizer from an optional config dict.
    If config is None, uses all defaults (Phase 3 spec).

    Args:
        config : Optional dict with keys matching PositionSizer __init__ params.

    Returns:
        PositionSizer instance ready to use.
    """
    defaults = {
        "kelly_fraction" : KELLY_FRACTION_BASE,
        "min_pct"        : MIN_POSITION_PCT,
        "max_pct"        : MAX_POSITION_PCT,
        "max_heat"       : MAX_PORTFOLIO_HEAT,
        "min_invest_inr" : MIN_INVEST_INR,
    }
    if config:
        defaults.update(config)
    return PositionSizer(**defaults)


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest environment/position_sizer.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestPositionSizer:
    """Unit tests for PositionSizer and related classes."""

    def _make_sizer(self) -> PositionSizer:
        return PositionSizer()

    def _make_input(self, **kwargs) -> SizingInput:
        defaults = dict(
            symbol           = "RELIANCE",
            entry_price      = 2850.0,
            tp_price         = 2964.0,     # 4% TP
            sl_price         = 2807.25,    # 1.5% SL
            confidence_score = 0.72,
            portfolio_value  = 1_000_000.0,
            available_cash   = 800_000.0,
            current_drawdown = 0.0,
            open_risk_pct    = 0.0,
            mds_score        = 0,
            vol_regime       = "normal",
            atr_pct          = 0.018,
            is_strong_signal = False,
        )
        defaults.update(kwargs)
        return SizingInput(**defaults)

    # ── Basic output tests ────────────────────────────────────────────────

    def test_returns_sizing_result(self):
        sizer = self._make_sizer()
        result = sizer.compute(self._make_input())
        assert isinstance(result, SizingResult)

    def test_normal_signal_is_not_zero(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input())
        assert not result.is_zero, f"Normal signal should not be zero: {result.reason}"

    def test_quantity_is_positive(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input())
        assert result.quantity >= 1

    def test_invest_amount_positive(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input())
        assert result.invest_amount > 0

    def test_position_pct_within_bounds(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input())
        assert MIN_POSITION_PCT <= result.position_pct <= MAX_POSITION_PCT, \
            f"position_pct {result.position_pct:.2%} out of bounds"

    def test_invest_amount_not_exceeding_cash(self):
        sizer  = self._make_sizer()
        inp    = self._make_input(available_cash=50_000.0)
        result = sizer.compute(inp)
        assert result.invest_amount <= inp.available_cash

    def test_quantity_times_price_equals_invest(self):
        """quantity × entry_price should equal invest_amount (within rounding)."""
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input())
        expected = result.quantity * 2850.0
        assert abs(result.invest_amount - expected) < 2850.0  # within 1 share

    # ── Kelly formula tests ───────────────────────────────────────────────

    def test_kelly_fraction_positive_for_good_trade(self):
        """Good trade (positive EV) must produce positive Kelly."""
        sizer = self._make_sizer()
        b, kelly, p = sizer._compute_kelly(self._make_input())
        assert kelly > 0, f"Kelly should be positive for good trade: {kelly}"

    def test_kelly_reward_risk_ratio_correct(self):
        """b = TP% / SL% = 4% / 1.5% ≈ 2.67."""
        sizer = self._make_sizer()
        inp   = self._make_input(
            entry_price=100.0, tp_price=104.0, sl_price=98.5
        )
        b, _, _ = sizer._compute_kelly(inp)
        assert abs(b - (4.0 / 1.5)) < 0.01, f"R:R ratio wrong: {b}"

    def test_kelly_capped_at_1(self):
        """Kelly fraction must never exceed 1.0."""
        sizer = self._make_sizer()
        inp   = self._make_input(
            tp_price=2850.0 * 2.0,    # 100% TP (extreme)
            confidence_score=0.99,
        )
        _, kelly, _ = sizer._compute_kelly(inp)
        assert kelly <= 1.0

    def test_kelly_floored_at_0(self):
        """Negative Kelly (bad EV trade) must return 0."""
        sizer = self._make_sizer()
        # Inverted TP/SL: tiny TP, huge SL = negative EV
        inp = self._make_input(
            entry_price=100.0,
            tp_price=100.5,    # 0.5% TP
            sl_price=90.0,     # 10% SL
            confidence_score=0.56,
        )
        _, kelly, _ = sizer._compute_kelly(inp)
        assert kelly == 0.0

    # ── Confidence scaling tests ──────────────────────────────────────────

    def test_high_confidence_larger_size(self):
        """High confidence should produce larger position than low confidence."""
        sizer     = self._make_sizer()
        high_conf = sizer.compute(self._make_input(confidence_score=0.85))
        low_conf  = sizer.compute(self._make_input(confidence_score=0.57))
        assert high_conf.position_pct > low_conf.position_pct, \
            "High confidence should give larger position"

    def test_confidence_below_threshold_is_zero(self):
        """Confidence below 0.55 must return is_zero=True."""
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input(confidence_score=0.45))
        assert result.is_zero
        assert "0.55" in result.reason

    def test_strong_signal_larger_than_normal(self):
        """Strong Buy signal should produce larger position than Buy."""
        sizer  = self._make_sizer()
        strong = sizer.compute(self._make_input(is_strong_signal=True))
        normal = sizer.compute(self._make_input(is_strong_signal=False))
        assert strong.position_pct >= normal.position_pct

    # ── MDS multiplier tests ──────────────────────────────────────────────

    def test_mds_plus3_increases_size(self):
        """MDS +3 should give larger position than MDS 0."""
        sizer    = self._make_sizer()
        bull     = sizer.compute(self._make_input(mds_score=3))
        neutral  = sizer.compute(self._make_input(mds_score=0))
        assert bull.position_pct > neutral.position_pct

    def test_mds_minus3_returns_zero(self):
        """MDS -3 = no new longs → is_zero=True."""
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input(mds_score=-3))
        assert result.is_zero
        assert "MDS" in result.reason

    def test_mds_minus2_reduces_size(self):
        """MDS -2 should produce smaller position than MDS 0."""
        sizer   = self._make_sizer()
        bearish = sizer.compute(self._make_input(mds_score=-2))
        neutral = sizer.compute(self._make_input(mds_score=0))
        if not bearish.is_zero:
            assert bearish.position_pct < neutral.position_pct

    def test_all_mds_values_valid(self):
        """All MDS values [-3, +3] must not crash."""
        sizer = self._make_sizer()
        for mds in range(-3, 4):
            result = sizer.compute(self._make_input(mds_score=mds))
            assert isinstance(result, SizingResult)

    # ── Volatility regime tests ───────────────────────────────────────────

    def test_high_vol_reduces_size(self):
        """High volatility should produce smaller position than normal."""
        sizer  = self._make_sizer()
        high_v = sizer.compute(self._make_input(vol_regime="high"))
        normal = sizer.compute(self._make_input(vol_regime="normal"))
        assert high_v.position_pct < normal.position_pct

    def test_low_vol_increases_size(self):
        """Low volatility should produce larger position than normal."""
        sizer  = self._make_sizer()
        low_v  = sizer.compute(self._make_input(vol_regime="low"))
        normal = sizer.compute(self._make_input(vol_regime="normal"))
        assert low_v.position_pct >= normal.position_pct

    def test_extreme_vol_smallest_size(self):
        """Extreme volatility should give the smallest position."""
        sizer   = self._make_sizer()
        extreme = sizer.compute(self._make_input(vol_regime="extreme"))
        normal  = sizer.compute(self._make_input(vol_regime="normal"))
        assert extreme.position_pct < normal.position_pct

    def test_extreme_atr_reduces_size(self):
        """Very high ATR% should reduce position size."""
        sizer    = self._make_sizer()
        high_atr = sizer.compute(self._make_input(atr_pct=0.05))
        norm_atr = sizer.compute(self._make_input(atr_pct=0.018))
        assert high_atr.position_pct <= norm_atr.position_pct

    # ── Drawdown protection tests ─────────────────────────────────────────

    def test_drawdown_reduces_size(self):
        """Significant drawdown should reduce position size."""
        sizer      = self._make_sizer()
        in_dd      = sizer.compute(self._make_input(current_drawdown=0.07))
        no_dd      = sizer.compute(self._make_input(current_drawdown=0.0))
        assert in_dd.position_pct < no_dd.position_pct

    def test_deep_drawdown_smallest_size(self):
        """Deep drawdown (>10%) should give minimum possible size."""
        sizer  = self._make_sizer()
        deep   = sizer.compute(self._make_input(current_drawdown=0.11))
        normal = sizer.compute(self._make_input(current_drawdown=0.0))
        assert deep.position_pct < normal.position_pct

    # ── Validation tests ──────────────────────────────────────────────────

    def test_zero_entry_price_returns_zero(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input(entry_price=0.0))
        assert result.is_zero

    def test_tp_below_entry_returns_zero(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input(tp_price=2800.0))  # below entry 2850
        assert result.is_zero

    def test_sl_above_entry_returns_zero(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input(sl_price=2900.0))  # above entry 2850
        assert result.is_zero

    def test_no_cash_returns_zero(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input(available_cash=100.0))
        assert result.is_zero

    def test_max_heat_reached_returns_zero(self):
        """At maximum portfolio heat, no new positions."""
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input(open_risk_pct=0.06))
        assert result.is_zero

    # ── Portfolio heat cap tests ──────────────────────────────────────────

    def test_heat_cap_limits_size(self):
        """Near-max heat should limit position size."""
        sizer     = self._make_sizer()
        near_heat = sizer.compute(self._make_input(open_risk_pct=0.04))
        no_heat   = sizer.compute(self._make_input(open_risk_pct=0.0))
        if not near_heat.is_zero:
            assert near_heat.position_pct <= no_heat.position_pct

    # ── Outcome tracking tests ────────────────────────────────────────────

    def test_record_outcome_updates_history(self):
        sizer = self._make_sizer()
        sizer.record_outcome(0.80, won=True)
        sizer.record_outcome(0.80, won=False)
        stats = sizer.get_overall_stats()
        assert stats["n_trades"]  == 2
        assert stats["win_rate"]  == 0.5

    def test_empirical_win_rates_insufficient_data(self):
        """With < 10 trades, empirical rates should return None."""
        sizer = self._make_sizer()
        sizer.record_outcome(0.80, won=True)
        rates = sizer.get_empirical_win_rates()
        assert rates["high"] is None

    def test_empirical_win_rates_sufficient_data(self):
        """With 10+ trades, empirical rates should be computed."""
        sizer = self._make_sizer()
        for _ in range(7):
            sizer.record_outcome(0.80, won=True)
        for _ in range(3):
            sizer.record_outcome(0.80, won=False)
        rates = sizer.get_empirical_win_rates()
        assert rates["high"] is not None
        assert abs(rates["high"] - 0.70) < 1e-6

    # ── SizingResult tests ────────────────────────────────────────────────

    def test_sizing_result_summary_no_crash(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input())
        summary = result.summary()
        assert isinstance(summary, str)
        assert "RELIANCE" in summary

    def test_zero_result_summary(self):
        sizer  = self._make_sizer()
        result = sizer.compute(self._make_input(mds_score=-3))
        assert result.is_zero
        summary = result.summary()
        assert "MDS" in summary

    # ── Factory test ──────────────────────────────────────────────────────

    def test_build_position_sizer_defaults(self):
        sizer = build_position_sizer()
        assert isinstance(sizer, PositionSizer)
        assert sizer.kelly_fraction == KELLY_FRACTION_BASE

    def test_build_position_sizer_custom(self):
        sizer = build_position_sizer({"kelly_fraction": 0.5})
        assert sizer.kelly_fraction == 0.5


# ── Run tests when file is executed directly ──────────────────────────────
if __name__ == "__main__":
    import sys
    import pytest

    # Quick demo
    sizer = PositionSizer()
    demo  = SizingInput(
        symbol="RELIANCE", entry_price=2850.0,
        tp_price=2964.0, sl_price=2807.25,
        confidence_score=0.78, portfolio_value=1_000_000,
        available_cash=800_000, current_drawdown=0.02,
        open_risk_pct=0.015, mds_score=1,
        vol_regime="normal", atr_pct=0.018,
    )
    result = sizer.compute(demo)
    print(result.summary())
    print()

    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))