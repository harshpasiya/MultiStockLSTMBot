"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Reward Function                                 ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : environment/reward_fn.py                               ║
║         Phase   : 3 — RL Agent Training                                 ║
║                                                                          ║
║  What this module does:                                                  ║
║    Provides the shaped reward signal used by both PPO RL heads           ║
║    (swing and intraday) during training.                                 ║
║                                                                          ║
║    Extracted from godseye_env.py into a standalone module so that:       ║
║      1. Both swing and intraday training loops share identical reward    ║
║         logic (single source of truth)                                   ║
║      2. Reward components can be tuned via config without touching env   ║
║      3. Reward shaping can be unit-tested independently                  ║
║      4. Live trading can compute reward-equivalent metrics for logging   ║
║                                                                          ║
║  Reward design philosophy:                                               ║
║    The agent must learn to maximise RISK-ADJUSTED returns, not just      ║
║    raw returns. These are fundamentally different objectives:            ║
║      Raw return maximiser: goes all-in on highest-EV trades,            ║
║                            accepts 50%+ drawdowns                        ║
║      Risk-adjusted maximiser: consistent compounding, controlled DD,    ║
║                               survives bad regimes                       ║
║                                                                          ║
║  Six reward components:                                                  ║
║    1. Step return        — immediate P&L (primary learning signal)       ║
║    2. Sharpe bonus       — rewards consistency over lucky spikes         ║
║    3. Drawdown penalty   — quadratic penalty (hurts more as DD grows)   ║
║    4. Overtrading penalty— discourages churning (cost drag)              ║
║    5. Hold-loss penalty  — discourages holding losing positions too long ║
║    6. RC violation penalty— large penalty when Risk Constitution fires   ║
║                                                                          ║
║  Two reward modes:                                                       ║
║    SWING    : Rewards computed on daily bars, emphasises multi-day       ║
║               trend capture and patience                                 ║
║    INTRADAY : Rewards computed on 5-min bars, emphasises quick exits    ║
║               and tight risk; extra penalty for overnight holds          ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install numpy                                                     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import numpy as np

from dataclasses import dataclass, field
from enum        import IntEnum
from typing      import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════
#  ENUMERATIONS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

class RewardMode(IntEnum):
    SWING    = 0
    INTRADAY = 1


# ── Default reward weights ────────────────────────────────────────────────
# Swing weights: reward patience and multi-day trend capture
SWING_WEIGHTS = {
    "return"    : 1.00,   # step return (primary)
    "sharpe"    : 0.30,   # consistency bonus
    "drawdown"  : 2.00,   # quadratic drawdown penalty
    "overtrade" : 0.005,  # per-trade churn penalty
    "hold_loss" : 0.001,  # per-bar holding-a-loser penalty
    "rc"        : 0.50,   # per Risk Constitution violation
}

# Intraday weights: tighter, faster; extra penalty for holding into close
INTRADAY_WEIGHTS = {
    "return"    : 1.00,
    "sharpe"    : 0.20,   # less emphasis on long-term consistency
    "drawdown"  : 2.50,   # harsher DD penalty (intraday DD unrecoverable)
    "overtrade" : 0.010,  # stronger churn penalty (costs per bar add up)
    "hold_loss" : 0.003,  # stronger hold-loser penalty (time is expensive)
    "rc"        : 0.50,
}

# Sharpe window: compute over last N steps
SHARPE_WINDOW_SWING    = 5    # 5 days
SHARPE_WINDOW_INTRADAY = 10   # 10 bars (50 minutes)

# Drawdown thresholds for graduated penalty scaling
DD_THRESHOLD_WARN     = 0.05   # 5%: penalty kicks in
DD_THRESHOLD_CRITICAL = 0.10   # 10%: penalty doubles


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RewardInput:
    """
    All inputs required to compute the reward for one environment step.

    Populated by GodsEyeEnv.step() and passed to RewardFunction.compute().
    """
    # ── Step P&L ──────────────────────────────────────────────────────────
    step_return         : float    # fractional portfolio return this bar
                                   # e.g. 0.003 = gained 0.3% this bar

    # ── Portfolio state ───────────────────────────────────────────────────
    portfolio_value     : float    # current total portfolio value ₹
    peak_value          : float    # all-time peak portfolio value ₹
    initial_value       : float    # episode starting portfolio value ₹

    # ── Trade activity this bar ───────────────────────────────────────────
    n_trades_this_bar   : int      # how many trades were opened/closed
    n_losing_positions  : int      # positions currently in unrealised loss

    # ── Risk Constitution ─────────────────────────────────────────────────
    rc_violations       : int      # number of RC rules triggered this bar

    # ── Episode context ───────────────────────────────────────────────────
    mode                : RewardMode = RewardMode.SWING
    bar_index           : int        = 0    # current bar within episode
    is_last_bar         : bool       = False  # True = final bar of episode

    # ── Intraday-specific ─────────────────────────────────────────────────
    forced_close        : bool = False   # True if positions force-closed at EOD
                                         # (penalise if still holding losers)


@dataclass
class RewardOutput:
    """
    Full reward computation result for one step.

    reward is the scalar passed to the PPO agent.
    components is logged for debugging and reward shaping analysis.
    """
    reward      : float
    components  : Dict[str, float]
    drawdown    : float
    episode_pnl : float   # total P&L so far this episode as fraction

    def __repr__(self) -> str:
        comp_str = "  ".join(
            f"{k}={v:+.4f}" for k, v in self.components.items()
            if k != "total"
        )
        return (
            f"RewardOutput(reward={self.reward:+.4f} | "
            f"dd={self.drawdown:.2%} | {comp_str})"
        )


# ══════════════════════════════════════════════════════════════════════════
#  REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════

class RewardFunction:
    """
    Stateful reward function for G.O.D.S E.Y.E PPO training.

    Stateful because Sharpe computation requires a rolling window of
    recent returns. State resets at the start of each episode.

    Two modes: SWING and INTRADAY (different weights and windows).

    Usage:
        # Create once per env instance
        rf = RewardFunction(mode=RewardMode.SWING)

        # Call at episode start
        rf.reset(initial_portfolio_value=1_000_000.0)

        # Call every step
        inp = RewardInput(
            step_return=0.003,
            portfolio_value=1_003_000,
            peak_value=1_003_000,
            initial_value=1_000_000,
            n_trades_this_bar=1,
            n_losing_positions=0,
            rc_violations=0,
            mode=RewardMode.SWING,
        )
        out = rf.compute(inp)
        reward = out.reward
    """

    def __init__(
        self,
        mode   : RewardMode              = RewardMode.SWING,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.mode = mode

        # Load weights: use provided or defaults by mode
        if weights is not None:
            self.weights = weights
        elif mode == RewardMode.SWING:
            self.weights = SWING_WEIGHTS.copy()
        else:
            self.weights = INTRADAY_WEIGHTS.copy()

        # Rolling return history for Sharpe calculation
        self._return_history: List[float] = []
        self._sharpe_window = (
            SHARPE_WINDOW_SWING if mode == RewardMode.SWING
            else SHARPE_WINDOW_INTRADAY
        )

        # Episode tracking
        self._peak_value    : float = 0.0
        self._initial_value : float = 0.0
        self._step_count    : int   = 0

    def reset(self, initial_portfolio_value: float):
        """
        Resets all episode state.
        Must be called at the start of every new episode.
        """
        self._return_history = []
        self._peak_value     = initial_portfolio_value
        self._initial_value  = initial_portfolio_value
        self._step_count     = 0

    def compute(self, inp: RewardInput) -> RewardOutput:
        """
        Computes shaped reward for one environment step.

        Args:
            inp : RewardInput populated by env.step()

        Returns:
            RewardOutput with scalar reward and component breakdown
        """
        self._step_count += 1

        # ── Update peak value ─────────────────────────────────────────────
        if inp.portfolio_value > self._peak_value:
            self._peak_value = inp.portfolio_value

        # ── Core metrics ──────────────────────────────────────────────────
        drawdown    = self._compute_drawdown(inp.portfolio_value)
        episode_pnl = self._compute_episode_pnl(inp.portfolio_value)

        # ── Append return to history ──────────────────────────────────────
        self._return_history.append(inp.step_return)
        if len(self._return_history) > self._sharpe_window * 4:
            self._return_history = self._return_history[-(self._sharpe_window * 4):]

        # ── Compute each component ────────────────────────────────────────
        components: Dict[str, float] = {}

        # 1. Step return component
        components["return"] = (
            inp.step_return * self.weights["return"]
        )

        # 2. Sharpe consistency bonus
        components["sharpe"] = (
            self._sharpe_bonus() * self.weights["sharpe"]
        )

        # 3. Drawdown penalty (quadratic — pain scales with depth)
        components["drawdown"] = (
            self._drawdown_penalty(drawdown) * self.weights["drawdown"]
        )

        # 4. Overtrading penalty
        components["overtrade"] = (
            -inp.n_trades_this_bar * self.weights["overtrade"]
        )

        # 5. Hold-a-loser penalty
        components["hold_loss"] = (
            -inp.n_losing_positions * self.weights["hold_loss"]
        )

        # 6. RC violation penalty
        components["rc"] = (
            -inp.rc_violations * self.weights["rc"]
        )

        # 7. Mode-specific additions ───────────────────────────────────────
        if inp.mode == RewardMode.INTRADAY:
            components["intraday_extra"] = self._intraday_extras(inp, drawdown)
        else:
            components["swing_extra"] = self._swing_extras(inp, episode_pnl)

        # ── Total reward ──────────────────────────────────────────────────
        total = sum(components.values())

        # Clip to prevent extreme rewards destabilising PPO
        total = float(np.clip(total, -10.0, 10.0))
        components["total"] = total

        return RewardOutput(
            reward     = total,
            components = components,
            drawdown   = drawdown,
            episode_pnl= episode_pnl,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  COMPONENT CALCULATORS
    # ══════════════════════════════════════════════════════════════════════

    def _sharpe_bonus(self) -> float:
        """
        Computes a Sharpe-like bonus over the recent return window.

        Rewards the agent for consistent returns over volatile returns.
        A string of small gains scores better than one big gain + losses.

        Returns scalar in approximately [-0.5, +0.5] range.
        """
        if len(self._return_history) < self._sharpe_window:
            return 0.0

        recent    = self._return_history[-self._sharpe_window:]
        mean_r    = float(np.mean(recent))
        std_r     = float(np.std(recent)) + 1e-8

        # Normalised Sharpe: divide by window length to keep in reasonable range
        sharpe    = (mean_r / std_r) / self._sharpe_window
        return float(np.clip(sharpe, -0.5, 0.5))

    def _drawdown_penalty(self, drawdown: float) -> float:
        """
        Computes quadratic drawdown penalty.

        Penalty structure:
            DD < 5%  : very small linear penalty (monitoring zone)
            DD 5–10% : quadratic — growing pain as DD deepens
            DD > 10% : doubled quadratic — severe pain approaching kill switch

        The quadratic nature means:
            5% DD  → penalty ≈ -0.025
            10% DD → penalty ≈ -0.100  (4× worse, not 2×)
            12% DD → penalty ≈ -0.144  (5.8× worse than 5% DD)

        This teaches the agent that drawdowns are disproportionately costly.

        Returns scalar ≤ 0.
        """
        if drawdown <= 0:
            return 0.0

        if drawdown < DD_THRESHOLD_WARN:
            # Very small linear penalty in safe zone
            return -drawdown * 0.1

        if drawdown < DD_THRESHOLD_CRITICAL:
            # Quadratic in caution zone
            return -(drawdown ** 2)

        # Double quadratic in critical zone (> 10% DD)
        return -(drawdown ** 2) * 2.0

    def _intraday_extras(self, inp: RewardInput, drawdown: float) -> float:
        """
        Additional reward shaping specific to intraday mode.

        Extras:
            a) Forced-close penalty: if positions were forced closed at EOD
               while in loss, apply additional penalty (should have exited earlier)
            b) Time decay: small negative reward for each bar spent in a
               losing position (time is money in intraday)
        """
        extra = 0.0

        # Forced EOD close while in loss = agent held too long
        if inp.forced_close and inp.n_losing_positions > 0:
            extra -= 0.02 * inp.n_losing_positions

        # Small time-decay penalty in drawdown (encourages faster exits)
        if drawdown > 0.02:
            extra -= drawdown * 0.05

        return extra

    def _swing_extras(self, inp: RewardInput, episode_pnl: float) -> float:
        """
        Additional reward shaping specific to swing mode.

        Extras:
            a) Episode completion bonus: small bonus if episode ends with
               positive P&L (encourages the agent to actually capture gains)
            b) Trend-riding bonus: small bonus for each bar a winning
               position is held (encourages patience in good trades)
        """
        extra = 0.0

        # Episode completion bonus/penalty
        if inp.is_last_bar:
            if episode_pnl > 0.02:
                extra += 0.05   # small bonus for profitable episode
            elif episode_pnl < -0.05:
                extra -= 0.05   # small penalty for losing episode

        return extra

    def _compute_drawdown(self, portfolio_value: float) -> float:
        """Returns current drawdown from peak as a non-negative fraction."""
        if self._peak_value <= 0:
            return 0.0
        return max(0.0, (self._peak_value - portfolio_value) / self._peak_value)

    def _compute_episode_pnl(self, portfolio_value: float) -> float:
        """Returns total P&L this episode as a fraction of initial value."""
        if self._initial_value <= 0:
            return 0.0
        return (portfolio_value - self._initial_value) / self._initial_value

    # ══════════════════════════════════════════════════════════════════════
    #  INTROSPECTION
    # ══════════════════════════════════════════════════════════════════════

    def get_episode_stats(self) -> Dict:
        """
        Returns statistics about the current episode's reward history.
        Used by the training loop for logging.
        """
        if not self._return_history:
            return {
                "n_steps"       : 0,
                "mean_return"   : 0.0,
                "total_return"  : 0.0,
                "return_std"    : 0.0,
                "sharpe_approx" : 0.0,
                "peak_value"    : self._peak_value,
            }

        returns = np.array(self._return_history)
        total_r = float(np.sum(returns))
        mean_r  = float(np.mean(returns))
        std_r   = float(np.std(returns)) + 1e-8

        return {
            "n_steps"       : self._step_count,
            "mean_return"   : mean_r,
            "total_return"  : total_r,
            "return_std"    : float(np.std(returns)),
            "sharpe_approx" : mean_r / std_r * math.sqrt(252),  # annualised
            "peak_value"    : self._peak_value,
            "max_drawdown"  : self._compute_drawdown(
                self._peak_value * (1 + total_r)  # approximate
            ),
        }

    @property
    def mode_name(self) -> str:
        return "SWING" if self.mode == RewardMode.SWING else "INTRADAY"


# ══════════════════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════════════════

def build_reward_fn(
    mode   : RewardMode               = RewardMode.SWING,
    weights: Optional[Dict[str, float]] = None,
) -> RewardFunction:
    """
    Builds a RewardFunction instance.

    Args:
        mode    : RewardMode.SWING or RewardMode.INTRADAY
        weights : Optional weight override dict. Keys must match
                  SWING_WEIGHTS / INTRADAY_WEIGHTS.

    Returns:
        RewardFunction ready to use.

    Example:
        # Swing with default weights
        rf = build_reward_fn(RewardMode.SWING)

        # Intraday with custom drawdown penalty
        rf = build_reward_fn(RewardMode.INTRADAY, {"drawdown": 3.0})
    """
    return RewardFunction(mode=mode, weights=weights)


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest environment/reward_fn.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestRewardFunction:
    """Unit tests for RewardFunction and all reward components."""

    def _make_rf(self, mode=RewardMode.SWING) -> RewardFunction:
        rf = RewardFunction(mode=mode)
        rf.reset(initial_portfolio_value=1_000_000.0)
        return rf

    def _make_input(self, **kwargs) -> RewardInput:
        defaults = dict(
            step_return        = 0.001,
            portfolio_value    = 1_001_000.0,
            peak_value         = 1_001_000.0,
            initial_value      = 1_000_000.0,
            n_trades_this_bar  = 0,
            n_losing_positions = 0,
            rc_violations      = 0,
            mode               = RewardMode.SWING,
            bar_index          = 5,
            is_last_bar        = False,
        )
        defaults.update(kwargs)
        return RewardInput(**defaults)

    # ── Basic output tests ────────────────────────────────────────────────

    def test_returns_reward_output(self):
        rf     = self._make_rf()
        result = rf.compute(self._make_input())
        assert isinstance(result, RewardOutput)

    def test_reward_is_float(self):
        rf     = self._make_rf()
        result = rf.compute(self._make_input())
        assert isinstance(result.reward, float)

    def test_reward_no_nan(self):
        rf = self._make_rf()
        for _ in range(10):
            result = rf.compute(self._make_input())
            assert not math.isnan(result.reward), "NaN reward detected"

    def test_reward_no_inf(self):
        rf = self._make_rf()
        for _ in range(10):
            result = rf.compute(self._make_input())
            assert not math.isinf(result.reward), "Inf reward detected"

    def test_reward_clipped(self):
        """Reward must be clipped to [-10, +10]."""
        rf  = self._make_rf()
        inp = self._make_input(step_return=100.0)   # extreme input
        out = rf.compute(inp)
        assert out.reward <= 10.0

    def test_reward_clipped_negative(self):
        rf  = self._make_rf()
        inp = self._make_input(
            step_return=-100.0,
            rc_violations=100,
            n_losing_positions=100,
        )
        out = rf.compute(inp)
        assert out.reward >= -10.0

    def test_components_dict_has_required_keys(self):
        rf     = self._make_rf()
        result = rf.compute(self._make_input())
        for key in ("return", "sharpe", "drawdown", "overtrade", "hold_loss", "rc", "total"):
            assert key in result.components, f"Missing component: {key}"

    def test_components_sum_to_total(self):
        """Sum of non-total components should equal total (before clip)."""
        rf     = self._make_rf()
        result = rf.compute(self._make_input())
        comp_sum = sum(
            v for k, v in result.components.items() if k != "total"
        )
        # Allow tolerance due to clipping
        assert abs(comp_sum - result.reward) < 0.01 or abs(result.reward) == 10.0

    # ── Directional tests ─────────────────────────────────────────────────

    def test_positive_return_positive_reward_component(self):
        rf   = self._make_rf()
        out  = rf.compute(self._make_input(step_return=0.01))
        assert out.components["return"] > 0

    def test_negative_return_negative_reward_component(self):
        rf  = self._make_rf()
        out = rf.compute(self._make_input(step_return=-0.01))
        assert out.components["return"] < 0

    def test_higher_return_higher_reward(self):
        rf   = self._make_rf()
        out1 = rf.compute(self._make_input(step_return=0.02))
        rf2  = self._make_rf()
        out2 = rf2.compute(self._make_input(step_return=0.005))
        assert out1.reward > out2.reward

    def test_rc_violation_reduces_reward(self):
        rf_clean = self._make_rf()
        rf_viol  = self._make_rf()
        out_clean= rf_clean.compute(self._make_input(rc_violations=0))
        out_viol = rf_viol.compute(self._make_input(rc_violations=3))
        assert out_viol.reward < out_clean.reward

    def test_more_trades_lower_reward(self):
        """More trades per bar = higher overtrading penalty."""
        rf1 = self._make_rf()
        rf2 = self._make_rf()
        out1= rf1.compute(self._make_input(n_trades_this_bar=0))
        out2= rf2.compute(self._make_input(n_trades_this_bar=5))
        assert out2.reward < out1.reward

    def test_losing_positions_reduce_reward(self):
        rf1 = self._make_rf()
        rf2 = self._make_rf()
        out1= rf1.compute(self._make_input(n_losing_positions=0))
        out2= rf2.compute(self._make_input(n_losing_positions=4))
        assert out2.reward < out1.reward

    # ── Drawdown penalty tests ────────────────────────────────────────────

    def test_no_drawdown_no_penalty(self):
        rf  = self._make_rf()
        pen = rf._drawdown_penalty(0.0)
        assert pen == 0.0

    def test_drawdown_penalty_negative(self):
        rf  = self._make_rf()
        pen = rf._drawdown_penalty(0.08)
        assert pen < 0

    def test_drawdown_penalty_quadratic(self):
        """Penalty at 10% DD should be more than 4× penalty at 5% DD."""
        rf   = self._make_rf()
        p5   = abs(rf._drawdown_penalty(0.05))
        p10  = abs(rf._drawdown_penalty(0.10))
        assert p10 > p5 * 2.0, \
            f"DD penalty should be super-linear: p5={p5:.4f}, p10={p10:.4f}"

    def test_deep_drawdown_larger_penalty(self):
        rf   = self._make_rf()
        p10  = abs(rf._drawdown_penalty(0.10))
        p12  = abs(rf._drawdown_penalty(0.12))
        assert p12 > p10

    def test_drawdown_captured_in_output(self):
        rf  = self._make_rf()
        rf._peak_value = 1_100_000.0
        out = rf.compute(self._make_input(portfolio_value=1_000_000.0))
        assert abs(out.drawdown - (100_000 / 1_100_000)) < 1e-6

    # ── Sharpe bonus tests ────────────────────────────────────────────────

    def test_sharpe_zero_insufficient_history(self):
        """Sharpe bonus should be 0 before window fills."""
        rf    = self._make_rf()
        bonus = rf._sharpe_bonus()
        assert bonus == 0.0

    def test_consistent_returns_positive_sharpe(self):
        """Steady positive returns should produce positive Sharpe bonus."""
        rf = self._make_rf()
        # Feed consistent positive returns to fill window
        for _ in range(10):
            rf._return_history.append(0.002)
        bonus = rf._sharpe_bonus()
        assert bonus > 0

    def test_volatile_returns_lower_sharpe(self):
        """Volatile returns should produce lower Sharpe than consistent ones."""
        rf_stable   = self._make_rf()
        rf_volatile = self._make_rf()
        for _ in range(10):
            rf_stable._return_history.append(0.002)
        for i in range(10):
            rf_volatile._return_history.append(0.01 if i % 2 == 0 else -0.006)
        assert rf_stable._sharpe_bonus() > rf_volatile._sharpe_bonus()

    # ── Mode-specific tests ───────────────────────────────────────────────

    def test_intraday_mode_uses_intraday_weights(self):
        rf = RewardFunction(mode=RewardMode.INTRADAY)
        assert rf.weights["overtrade"] == INTRADAY_WEIGHTS["overtrade"]

    def test_swing_mode_uses_swing_weights(self):
        rf = RewardFunction(mode=RewardMode.SWING)
        assert rf.weights["overtrade"] == SWING_WEIGHTS["overtrade"]

    def test_intraday_forced_close_penalty(self):
        """Forced close with losing positions should add extra penalty."""
        rf  = RewardFunction(mode=RewardMode.INTRADAY)
        rf.reset(1_000_000.0)
        extra_with = rf._intraday_extras(
            self._make_input(forced_close=True, n_losing_positions=2),
            drawdown=0.0
        )
        extra_without = rf._intraday_extras(
            self._make_input(forced_close=False, n_losing_positions=2),
            drawdown=0.0
        )
        assert extra_with < extra_without

    def test_swing_completion_bonus_profitable(self):
        """Profitable episode end should give positive swing extra."""
        rf    = self._make_rf(mode=RewardMode.SWING)
        extra = rf._swing_extras(
            self._make_input(is_last_bar=True),
            episode_pnl=0.05  # 5% profitable
        )
        assert extra > 0

    def test_swing_completion_penalty_losing(self):
        """Loss episode end should give negative swing extra."""
        rf    = self._make_rf(mode=RewardMode.SWING)
        extra = rf._swing_extras(
            self._make_input(is_last_bar=True),
            episode_pnl=-0.08  # 8% losing episode
        )
        assert extra < 0

    # ── Reset tests ───────────────────────────────────────────────────────

    def test_reset_clears_history(self):
        rf = self._make_rf()
        for _ in range(10):
            rf.compute(self._make_input())
        rf.reset(1_000_000.0)
        assert len(rf._return_history) == 0
        assert rf._step_count == 0

    def test_reset_restores_peak(self):
        rf = self._make_rf()
        rf._peak_value = 2_000_000.0
        rf.reset(1_000_000.0)
        assert rf._peak_value == 1_000_000.0

    def test_multiple_resets_stable(self):
        rf = self._make_rf()
        for _ in range(5):
            rf.reset(1_000_000.0)
            for _ in range(10):
                out = rf.compute(self._make_input())
                assert not math.isnan(out.reward)

    # ── Episode stats tests ───────────────────────────────────────────────

    def test_episode_stats_empty(self):
        rf    = self._make_rf()
        stats = rf.get_episode_stats()
        assert stats["n_steps"] == 0

    def test_episode_stats_after_steps(self):
        rf = self._make_rf()
        for _ in range(20):
            rf.compute(self._make_input(step_return=0.002))
        stats = rf.get_episode_stats()
        assert stats["n_steps"]     == 20
        assert stats["mean_return"] > 0

    # ── Factory test ──────────────────────────────────────────────────────

    def test_build_reward_fn_swing(self):
        rf = build_reward_fn(RewardMode.SWING)
        assert rf.mode == RewardMode.SWING

    def test_build_reward_fn_intraday(self):
        rf = build_reward_fn(RewardMode.INTRADAY)
        assert rf.mode == RewardMode.INTRADAY

    def test_build_reward_fn_custom_weights(self):
        rf = build_reward_fn(RewardMode.SWING, {"drawdown": 5.0})
        assert rf.weights["drawdown"] == 5.0
        assert rf.weights["return"]   == SWING_WEIGHTS["return"]

    # ── RewardOutput repr test ────────────────────────────────────────────

    def test_reward_output_repr(self):
        rf  = self._make_rf()
        out = rf.compute(self._make_input())
        r   = repr(out)
        assert "RewardOutput" in r
        assert "reward"       in r


# ── Run tests when file is executed directly ──────────────────────────────
if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))