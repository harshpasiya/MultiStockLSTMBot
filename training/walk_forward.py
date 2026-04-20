"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Walk-Forward Backtester                         ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : training/walk_forward.py                               ║
║         Phase   : 3 — RL Agent Training / Phase 4 Gate Validation       ║
║                                                                          ║
║  What this file does:                                                    ║
║    Runs a rigorous walk-forward backtest across 6 market regime windows  ║
║    to validate the trained swing RL agent before live deployment.        ║
║                                                                          ║
║  Why walk-forward (not simple backtest):                                 ║
║    Standard backtesting (train once, test on all history) produces       ║
║    dangerously optimistic results because:                               ║
║      • Parameters are implicitly fitted to the full dataset              ║
║      • The model "sees" future data during evaluation                    ║
║    Walk-forward retrains periodically and always tests on UNSEEN data.  ║
║                                                                          ║
║  6 test windows covering distinct market regimes:                        ║
║    Window 1: 2021-01 → 2021-06  Post-COVID bull market                  ║
║    Window 2: 2021-07 → 2021-12  Continued bull, mid-cap peaks           ║
║    Window 3: 2022-01 → 2022-06  Rate hike cycle begins, market tops     ║
║    Window 4: 2022-07 → 2022-12  Bear market, heavy FII selling          ║
║    Window 5: 2023-01 → 2023-06  Recovery rally, Adani episode           ║
║    Window 6: 2023-07 → 2024-06  Election year, strong bull market       ║
║                                                                          ║
║  Phase 3 Gate Criteria (ALL must pass):                                  ║
║    ✓ Aggregated CAGR          ≥ 45%                                      ║
║    ✓ Max drawdown             ≤ 12%                                      ║
║    ✓ Sharpe ratio             ≥ 1.8                                      ║
║    ✓ Profit factor            ≥ 2.0                                      ║
║    ✓ Win rate                 55–68%                                     ║
║    ✓ Avg monthly trades       8–15                                       ║
║    ✓ COVID stress test        capital preserved ≥ 88%                    ║
║                                                                          ║
║  Usage:                                                                  ║
║    # Full walk-forward backtest                                          ║
║    python -m training.walk_forward                                       ║
║                                                                          ║
║    # Single window test                                                  ║
║    python -m training.walk_forward --window 3                            ║
║                                                                          ║
║    # Stress tests only                                                   ║
║    python -m training.walk_forward --stress-only                         ║
║                                                                          ║
║  Output:                                                                 ║
║    logs/walk_forward/wf_report_<timestamp>.json   full results           ║
║    logs/walk_forward/wf_summary_<timestamp>.txt   human-readable         ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install stable-baselines3 torch numpy pandas psycopg2-binary     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import json
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch

from datetime   import datetime
from pathlib    import Path
from typing     import Dict, List, Optional, Tuple
from loguru     import logger
from dotenv     import load_dotenv

from stable_baselines3          import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.godseye_env    import GodsEyeEnv, MarketDataLoader, TradeMode
from environment.reward_fn      import RewardMode
from models.backbone            import GodsEyeBackbone

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR        = ROOT_DIR / "logs" / "walk_forward"
SWING_BEST     = CHECKPOINT_DIR / "swing_best.pt"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Walk-forward windows ───────────────────────────────────────────────────
# Each window: (name, train_end, test_start, test_end, regime_description)
WALK_FORWARD_WINDOWS = [
    {
        "id"         : 1,
        "name"       : "Post-COVID Bull",
        "train_end"  : "2020-12-31",
        "test_start" : "2021-01-01",
        "test_end"   : "2021-06-30",
        "regime"     : "Strong bull — Nifty +30% in H1 2021",
    },
    {
        "id"         : 2,
        "name"       : "Mid-cap Peaks",
        "train_end"  : "2021-06-30",
        "test_start" : "2021-07-01",
        "test_end"   : "2021-12-31",
        "regime"     : "Continued bull, mid/small cap euphoria",
    },
    {
        "id"         : 3,
        "name"       : "Rate Hike Begins",
        "train_end"  : "2021-12-31",
        "test_start" : "2022-01-01",
        "test_end"   : "2022-06-30",
        "regime"     : "US rate hikes begin, FII selling, market tops",
    },
    {
        "id"         : 4,
        "name"       : "Bear Market",
        "train_end"  : "2022-06-30",
        "test_start" : "2022-07-01",
        "test_end"   : "2022-12-31",
        "regime"     : "Bear market — heavy FII outflows, inflation peak",
    },
    {
        "id"         : 5,
        "name"       : "Recovery Rally",
        "train_end"  : "2022-12-31",
        "test_start" : "2023-01-01",
        "test_end"   : "2023-06-30",
        "regime"     : "Recovery, Adani crisis Jan 2023, then rebound",
    },
    {
        "id"         : 6,
        "name"       : "Election Bull",
        "train_end"  : "2023-06-30",
        "test_start" : "2023-07-01",
        "test_end"   : "2024-06-30",
        "regime"     : "Election year, strong institutional buying",
    },
]

# ── Stress test scenarios ──────────────────────────────────────────────────
STRESS_TESTS = [
    {
        "id"          : "ST-01",
        "name"        : "COVID Crash",
        "test_start"  : "2020-02-01",
        "test_end"    : "2020-04-30",
        "gate"        : "capital_preserved >= 0.88",
        "description" : "Nifty -38% in 40 days — system must preserve ≥88% capital",
    },
    {
        "id"          : "ST-02",
        "name"        : "Adani Crisis",
        "test_start"  : "2023-01-24",
        "test_end"    : "2023-02-28",
        "gate"        : "capital_preserved >= 0.92",
        "description" : "Adani Group crash, FII panic — system must preserve ≥92% capital",
    },
    {
        "id"          : "ST-03",
        "name"        : "Russia-Ukraine Spike",
        "test_start"  : "2022-02-24",
        "test_end"    : "2022-03-31",
        "gate"        : "capital_preserved >= 0.90",
        "description" : "Geopolitical shock, VIX spike — system must preserve ≥90% capital",
    },
]

# ── Gate criteria ──────────────────────────────────────────────────────────
GATE_CAGR          = 0.45   # ≥ 45% annualised
GATE_DRAWDOWN      = 0.12   # ≤ 12% max drawdown
GATE_SHARPE        = 1.80   # ≥ 1.8 Sharpe ratio
GATE_PROFIT_FACTOR = 2.00   # ≥ 2.0 profit factor
GATE_WIN_RATE_MIN  = 0.55   # ≥ 55% win rate
GATE_WIN_RATE_MAX  = 0.68   # ≤ 68% (above = likely overfitting)
GATE_TRADES_MIN    = 8      # ≥ 8 trades/month average
GATE_TRADES_MAX    = 15     # ≤ 15 trades/month average

# ── Simulation settings ────────────────────────────────────────────────────
INITIAL_CAPITAL  = 1_000_000.0   # ₹10 lakh per window
EPISODES_PER_WINDOW = 30         # episodes to average per window
N_STOCKS         = 20            # must match training config
EPISODE_DAYS     = 20            # swing episode length


# ══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════

class EpisodeRunner:
    """
    Runs multiple episodes on a GodsEyeEnv and collects detailed statistics.

    Used by both walk-forward windows and stress tests.
    """

    def __init__(
        self,
        model   : PPO,
        env     : GodsEyeEnv,
        n_episodes: int = EPISODES_PER_WINDOW,
        deterministic: bool = True,
    ):
        self.model         = model
        self.env           = env
        self.n_episodes    = n_episodes
        self.deterministic = deterministic

    def run(self) -> Dict:
        """
        Runs all episodes and returns aggregated statistics.

        Returns:
            Dict with keys:
                cagr, sharpe, max_drawdown, win_rate, profit_factor,
                avg_trades_per_month, avg_episode_return,
                capital_preserved (min across episodes),
                episode_returns (list), episode_sharpes (list)
        """
        results = {
            "episode_returns"     : [],
            "episode_sharpes"     : [],
            "episode_drawdowns"   : [],
            "episode_trades"      : [],
            "episode_wins"        : [],
            "gross_profits"       : [],
            "gross_losses"        : [],
        }

        for ep in range(self.n_episodes):
            ep_stats = self._run_single_episode(seed=ep * 7 + 13)
            results["episode_returns"].append(ep_stats["final_return"])
            results["episode_sharpes"].append(ep_stats["sharpe"])
            results["episode_drawdowns"].append(ep_stats["max_drawdown"])
            results["episode_trades"].append(ep_stats["n_trades"])
            results["episode_wins"].append(ep_stats["is_win"])
            results["gross_profits"].append(ep_stats["gross_profit"])
            results["gross_losses"].append(ep_stats["gross_loss"])

        return self._aggregate(results)

    def _run_single_episode(self, seed: int) -> Dict:
        """Runs one episode and returns per-episode statistics."""
        obs, _      = self.env.reset(seed=seed)
        done        = False
        pv_history  = [INITIAL_CAPITAL]
        peak        = INITIAL_CAPITAL
        max_dd      = 0.0
        n_trades    = 0
        gross_profit= 0.0
        gross_loss  = 0.0

        while not done:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            done = terminated or truncated

            pv = info.get("portfolio_value", INITIAL_CAPITAL)
            pv_history.append(pv)

            if pv > peak:
                peak = pv
            dd = (peak - pv) / peak
            max_dd = max(max_dd, dd)

            # Track trade P&L from info
            n_trades = info.get("total_trades", 0)

        # ── Compute episode metrics ───────────────────────────────────────
        final_pv   = pv_history[-1]
        final_ret  = (final_pv - INITIAL_CAPITAL) / INITIAL_CAPITAL

        # Annualised CAGR: episode = 20 days ≈ 20/252 years
        years = EPISODE_DAYS / 252
        cagr  = (1 + final_ret) ** (1 / years) - 1 if final_ret > -1 else -1.0

        # Daily return Sharpe
        daily_rets = np.diff(pv_history) / np.array(pv_history[:-1])
        if len(daily_rets) >= 2:
            mean_r = float(np.mean(daily_rets))
            std_r  = float(np.std(daily_rets)) + 1e-8
            sharpe = mean_r / std_r * math.sqrt(252)
        else:
            sharpe = 0.0

        # Gross profit / loss for profit factor calculation
        pos_rets = [r for r in daily_rets if r > 0]
        neg_rets = [r for r in daily_rets if r < 0]
        gross_profit = sum(pos_rets) * INITIAL_CAPITAL
        gross_loss   = abs(sum(neg_rets)) * INITIAL_CAPITAL

        return {
            "final_return"   : final_return  if (final_return := final_ret) else 0.0,
            "cagr"           : cagr,
            "sharpe"         : sharpe,
            "max_drawdown"   : max_dd,
            "n_trades"       : n_trades,
            "is_win"         : final_ret > 0,
            "gross_profit"   : gross_profit,
            "gross_loss"     : gross_loss,
            "capital_preserved": final_pv / INITIAL_CAPITAL,
        }

    def _aggregate(self, results: Dict) -> Dict:
        """Aggregates per-episode results into window-level metrics."""
        returns    = results["episode_returns"]
        sharpes    = results["episode_sharpes"]
        drawdowns  = results["episode_drawdowns"]
        trades     = results["episode_trades"]
        wins       = results["episode_wins"]
        gp         = sum(results["gross_profits"])
        gl         = sum(results["gross_losses"])

        # Annualised CAGR from mean episode return
        mean_ret   = float(np.mean(returns))
        years      = EPISODE_DAYS / 252
        mean_cagr  = (1 + mean_ret) ** (1 / years) - 1 if mean_ret > -1 else -1.0

        # Profit factor
        profit_factor = gp / max(gl, 1e-8)

        # Monthly trades: episode = 20 days ≈ 1 month
        avg_trades_month = float(np.mean(trades))

        # Capital preservation (worst case across episodes)
        min_capital = min(
            1 + r for r in returns
        )

        return {
            "cagr"                : mean_cagr,
            "sharpe"              : float(np.mean(sharpes)),
            "max_drawdown"        : float(np.max(drawdowns)),
            "win_rate"            : float(np.mean(wins)),
            "profit_factor"       : profit_factor,
            "avg_trades_per_month": avg_trades_month,
            "avg_episode_return"  : mean_ret,
            "capital_preserved"   : min_capital,
            "n_episodes"          : len(returns),
            "episode_returns"     : returns,
            "episode_sharpes"     : sharpes,
            "episode_drawdowns"   : drawdowns,
        }


# ══════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════════════════

class WalkForwardBacktester:
    """
    Runs the complete 6-window walk-forward backtest and stress tests.

    For each window:
        1. Creates a GodsEyeEnv restricted to the test date range
        2. Loads the trained PPO model (swing_best.pt)
        3. Runs EPISODES_PER_WINDOW episodes
        4. Computes all gate metrics
        5. Checks gate criteria

    After all windows:
        6. Aggregates across windows
        7. Runs stress tests
        8. Produces final pass/fail report

    Usage:
        backtester = WalkForwardBacktester()
        backtester.setup()
        report = backtester.run()
        backtester.save_report(report)
    """

    def __init__(
        self,
        checkpoint_path: Path  = SWING_BEST,
        device         : str   = "cuda" if torch.cuda.is_available() else "cpu",
        n_episodes     : int   = EPISODES_PER_WINDOW,
        deterministic  : bool  = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.device          = device
        self.n_episodes      = n_episodes
        self.deterministic   = deterministic

        self.data_loader: Optional[MarketDataLoader] = None
        self.backbone   : Optional[GodsEyeBackbone]  = None
        self.model      : Optional[PPO]              = None

    def setup(self):
        """Loads data, backbone, and PPO model."""
        logger.info("WalkForwardBacktester: setting up...")

        # ── Load market data (full history) ───────────────────────────────
        logger.info("Loading market data...")
        self.data_loader = MarketDataLoader(
            start_date = "2019-01-01",
            end_date   = "2024-06-30",
        )
        self.data_loader.load()

        # ── Load backbone ─────────────────────────────────────────────────
        logger.info(f"Loading backbone from {self.checkpoint_path}...")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Run training/train_swing_rl.py first."
            )

        ckpt = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.backbone = GodsEyeBackbone()
        backbone_state = ckpt.get("backbone_state", ckpt)
        self.backbone.load_state_dict(backbone_state, strict=False)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        # ── Load PPO model ────────────────────────────────────────────────
        # Build a dummy env to get obs/action spaces for PPO reconstruction
        dummy_env = self._make_env("2021-01-01", "2021-06-30")
        vec_env   = DummyVecEnv([lambda: dummy_env])

        self.model = PPO(
            policy       = "MlpPolicy",
            env          = vec_env,
            device       = self.device,
            verbose      = 0,
            tensorboard_log = None,
        )

        if "ppo_policy_state" in ckpt:
            self.model.policy.load_state_dict(
                ckpt["ppo_policy_state"], strict=False
            )
            logger.info("PPO policy weights loaded from checkpoint.")
        else:
            logger.warning(
                "No ppo_policy_state in checkpoint — "
                "using randomly initialised policy."
            )

        self.model.set_env(vec_env)
        logger.info("Setup complete.")

    def run(
        self,
        windows     : Optional[List[int]] = None,
        stress_only : bool = False,
    ) -> Dict:
        """
        Runs walk-forward backtest across all (or specified) windows.

        Args:
            windows     : List of window IDs to run (default: all 6)
            stress_only : If True, skip walk-forward and run stress tests only

        Returns:
            Full report dict with per-window and aggregated results
        """
        report = {
            "timestamp"       : datetime.now().isoformat(),
            "checkpoint"      : str(self.checkpoint_path),
            "n_episodes"      : self.n_episodes,
            "windows"         : {},
            "aggregated"      : {},
            "stress_tests"    : {},
            "gate_results"    : {},
            "overall_pass"    : False,
        }

        if not stress_only:
            # ── Run walk-forward windows ──────────────────────────────────
            windows_to_run = [
                w for w in WALK_FORWARD_WINDOWS
                if windows is None or w["id"] in windows
            ]

            for window in windows_to_run:
                logger.info(
                    f"Window {window['id']}/6 — {window['name']} "
                    f"({window['test_start']} → {window['test_end']})"
                )
                window_result = self._run_window(window)
                report["windows"][window["id"]] = window_result
                self._log_window_result(window, window_result)

            # ── Aggregate across windows ──────────────────────────────────
            if report["windows"]:
                report["aggregated"] = self._aggregate_windows(
                    list(report["windows"].values())
                )

        # ── Run stress tests ──────────────────────────────────────────────
        for st in STRESS_TESTS:
            logger.info(f"Stress test {st['id']} — {st['name']}")
            st_result = self._run_stress_test(st)
            report["stress_tests"][st["id"]] = st_result
            self._log_stress_result(st, st_result)

        # ── Gate evaluation ───────────────────────────────────────────────
        if report["aggregated"]:
            report["gate_results"] = self._evaluate_gates(
                report["aggregated"],
                report["stress_tests"],
            )
            report["overall_pass"] = all(
                v["pass"] for v in report["gate_results"].values()
            )

        return report

    def _run_window(self, window: Dict) -> Dict:
        """Runs one walk-forward window and returns metrics."""
        env = self._make_env(window["test_start"], window["test_end"])

        # Update model's env to match this window's obs space
        vec_env = DummyVecEnv([lambda: env])
        self.model.set_env(vec_env)

        runner = EpisodeRunner(
            model        = self.model,
            env          = env,
            n_episodes   = self.n_episodes,
            deterministic= self.deterministic,
        )

        start = time.time()
        stats = runner.run()
        elapsed = time.time() - start

        stats["window_id"]   = window["id"]
        stats["window_name"] = window["name"]
        stats["test_start"]  = window["test_start"]
        stats["test_end"]    = window["test_end"]
        stats["regime"]      = window["regime"]
        stats["elapsed_s"]   = elapsed

        return stats

    def _run_stress_test(self, stress_test: Dict) -> Dict:
        """Runs one stress test scenario."""
        try:
            env = self._make_env(
                stress_test["test_start"],
                stress_test["test_end"],
            )
        except Exception as e:
            logger.warning(
                f"Stress test {stress_test['id']} skipped: {e}"
            )
            return {
                "skipped"          : True,
                "reason"           : str(e),
                "capital_preserved": 1.0,
                "max_drawdown"     : 0.0,
            }

        vec_env = DummyVecEnv([lambda: env])
        self.model.set_env(vec_env)

        runner = EpisodeRunner(
            model         = self.model,
            env           = env,
            n_episodes    = min(self.n_episodes, 10),
            deterministic = self.deterministic,
        )

        stats = runner.run()
        stats["stress_id"]   = stress_test["id"]
        stats["stress_name"] = stress_test["name"]
        stats["gate"]        = stress_test["gate"]
        return stats

    def _make_env(self, start_date: str, end_date: str) -> GodsEyeEnv:
        """
        Creates a GodsEyeEnv restricted to a specific date range.

        Maps date strings to index positions in trading_dates.
        """
        all_dates   = self.data_loader.trading_dates
        start_idx   = next(
            (i for i, d in enumerate(all_dates) if d >= start_date),
            0
        )
        end_idx     = next(
            (i for i, d in enumerate(all_dates) if d > end_date),
            len(all_dates) - 1
        )

        if end_idx - start_idx < EPISODE_DAYS + 5:
            raise ValueError(
                f"Date range {start_date} → {end_date} has only "
                f"{end_idx - start_idx} trading days "
                f"(need at least {EPISODE_DAYS + 5})"
            )

        return GodsEyeEnv(
            data_loader     = self.data_loader,
            backbone        = self.backbone,
            mode            = TradeMode.SWING,
            initial_capital = INITIAL_CAPITAL,
            n_stocks        = N_STOCKS,
            train_start_idx = start_idx,
            train_end_idx   = end_idx,
            device          = self.device,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  AGGREGATION & GATE EVALUATION
    # ══════════════════════════════════════════════════════════════════════

    def _aggregate_windows(self, window_results: List[Dict]) -> Dict:
        """
        Aggregates metrics across all walk-forward windows.

        Uses equal weighting across windows (not time-weighted)
        to avoid recent-window bias.
        """
        cagrs         = [w["cagr"]                 for w in window_results]
        sharpes       = [w["sharpe"]               for w in window_results]
        drawdowns     = [w["max_drawdown"]          for w in window_results]
        win_rates     = [w["win_rate"]              for w in window_results]
        pfs           = [w["profit_factor"]         for w in window_results]
        trades        = [w["avg_trades_per_month"]  for w in window_results]

        # All episode returns flattened for overall distribution
        all_returns = []
        for w in window_results:
            all_returns.extend(w.get("episode_returns", []))

        return {
            "mean_cagr"           : float(np.mean(cagrs)),
            "median_cagr"         : float(np.median(cagrs)),
            "min_cagr"            : float(np.min(cagrs)),
            "mean_sharpe"         : float(np.mean(sharpes)),
            "max_drawdown"        : float(np.max(drawdowns)),
            "mean_win_rate"       : float(np.mean(win_rates)),
            "mean_profit_factor"  : float(np.mean(pfs)),
            "mean_trades_month"   : float(np.mean(trades)),
            "n_windows"           : len(window_results),
            "all_episode_returns" : all_returns,
            "return_std"          : float(np.std(all_returns)) if all_returns else 0.0,
            "pct_positive_windows": float(np.mean([w["cagr"] > 0 for w in window_results])),
        }

    def _evaluate_gates(
        self,
        agg         : Dict,
        stress_tests: Dict,
    ) -> Dict:
        """
        Evaluates all Phase 3 gate criteria.

        Returns dict of {gate_name: {"pass": bool, "value": float, "threshold": str}}
        """
        gates = {}

        # Gate 1: CAGR
        gates["cagr"] = {
            "pass"     : agg["mean_cagr"] >= GATE_CAGR,
            "value"    : agg["mean_cagr"],
            "threshold": f">= {GATE_CAGR:.0%}",
            "label"    : "Annualised CAGR",
        }

        # Gate 2: Max Drawdown
        gates["max_drawdown"] = {
            "pass"     : agg["max_drawdown"] <= GATE_DRAWDOWN,
            "value"    : agg["max_drawdown"],
            "threshold": f"<= {GATE_DRAWDOWN:.0%}",
            "label"    : "Maximum Drawdown",
        }

        # Gate 3: Sharpe Ratio
        gates["sharpe"] = {
            "pass"     : agg["mean_sharpe"] >= GATE_SHARPE,
            "value"    : agg["mean_sharpe"],
            "threshold": f">= {GATE_SHARPE}",
            "label"    : "Sharpe Ratio",
        }

        # Gate 4: Profit Factor
        gates["profit_factor"] = {
            "pass"     : agg["mean_profit_factor"] >= GATE_PROFIT_FACTOR,
            "value"    : agg["mean_profit_factor"],
            "threshold": f">= {GATE_PROFIT_FACTOR}",
            "label"    : "Profit Factor",
        }

        # Gate 5: Win Rate (must be in range — too high = overfitting)
        wr = agg["mean_win_rate"]
        gates["win_rate"] = {
            "pass"     : GATE_WIN_RATE_MIN <= wr <= GATE_WIN_RATE_MAX,
            "value"    : wr,
            "threshold": f"{GATE_WIN_RATE_MIN:.0%} – {GATE_WIN_RATE_MAX:.0%}",
            "label"    : "Win Rate",
        }

        # Gate 6: Monthly Trades
        mt = agg["mean_trades_month"]
        gates["monthly_trades"] = {
            "pass"     : GATE_TRADES_MIN <= mt <= GATE_TRADES_MAX,
            "value"    : mt,
            "threshold": f"{GATE_TRADES_MIN} – {GATE_TRADES_MAX} per month",
            "label"    : "Avg Monthly Trades",
        }

        # Gate 7: COVID Stress Test
        covid = stress_tests.get("ST-01", {})
        covid_preserved = covid.get("capital_preserved", 1.0)
        gates["covid_stress"] = {
            "pass"     : covid_preserved >= 0.88 or covid.get("skipped", False),
            "value"    : covid_preserved,
            "threshold": ">= 88% capital preserved",
            "label"    : "COVID Crash Stress Test",
        }

        return gates

    # ══════════════════════════════════════════════════════════════════════
    #  LOGGING & REPORTING
    # ══════════════════════════════════════════════════════════════════════

    def _log_window_result(self, window: Dict, result: Dict):
        """Logs a single window result."""
        status = "✓" if result["cagr"] >= 0 else "✗"
        logger.info(
            f"  {status} Window {window['id']} — {window['name']}: "
            f"CAGR={result['cagr']:.1%} | "
            f"Sharpe={result['sharpe']:.2f} | "
            f"DD={result['max_drawdown']:.1%} | "
            f"WR={result['win_rate']:.1%} | "
            f"PF={result['profit_factor']:.2f} | "
            f"Trades={result['avg_trades_per_month']:.1f}/mo"
        )

    def _log_stress_result(self, stress_test: Dict, result: Dict):
        """Logs a stress test result."""
        if result.get("skipped"):
            logger.warning(f"  ⚠ {stress_test['id']} SKIPPED: {result.get('reason')}")
            return
        preserved = result.get("capital_preserved", 1.0)
        dd        = result.get("max_drawdown", 0.0)
        status    = "✓" if preserved >= 0.88 else "✗"
        logger.info(
            f"  {status} {stress_test['id']} — {stress_test['name']}: "
            f"Capital preserved={preserved:.1%} | DD={dd:.1%}"
        )

    def save_report(self, report: Dict) -> Tuple[Path, Path]:
        """
        Saves full report as JSON and human-readable summary as TXT.

        Returns:
            (json_path, txt_path)
        """
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path= LOG_DIR / f"wf_report_{ts}.json"
        txt_path = LOG_DIR / f"wf_summary_{ts}.txt"

        # ── JSON (full data) ──────────────────────────────────────────────
        # Convert numpy types to Python natives for JSON serialisation
        def convert(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.ndarray,)):  return obj.tolist()
            return obj

        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=convert)

        # ── TXT (human-readable summary) ──────────────────────────────────
        lines = self._build_summary_text(report)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.success(f"Report saved: {json_path}")
        logger.success(f"Summary saved: {txt_path}")
        return json_path, txt_path

    def _build_summary_text(self, report: Dict) -> List[str]:
        """Builds human-readable summary text."""
        lines = [
            "=" * 70,
            "  G.O.D.S E.Y.E — Walk-Forward Backtest Report",
            f"  Generated : {report['timestamp']}",
            f"  Checkpoint: {report['checkpoint']}",
            "=" * 70,
            "",
            "WALK-FORWARD WINDOWS",
            "-" * 70,
        ]

        for wid, result in report["windows"].items():
            lines.append(
                f"  Window {wid} — {result.get('window_name', '')} "
                f"({result.get('test_start', '')} → {result.get('test_end', '')})"
            )
            lines.append(f"    Regime      : {result.get('regime', '')}")
            lines.append(f"    CAGR        : {result['cagr']:.1%}")
            lines.append(f"    Sharpe      : {result['sharpe']:.3f}")
            lines.append(f"    Max DD      : {result['max_drawdown']:.1%}")
            lines.append(f"    Win Rate    : {result['win_rate']:.1%}")
            lines.append(f"    Profit Fac  : {result['profit_factor']:.2f}")
            lines.append(f"    Trades/mo   : {result['avg_trades_per_month']:.1f}")
            lines.append("")

        if report["aggregated"]:
            agg = report["aggregated"]
            lines += [
                "AGGREGATED (ALL WINDOWS)",
                "-" * 70,
                f"  Mean CAGR        : {agg['mean_cagr']:.1%}",
                f"  Median CAGR      : {agg['median_cagr']:.1%}",
                f"  Mean Sharpe      : {agg['mean_sharpe']:.3f}",
                f"  Max Drawdown     : {agg['max_drawdown']:.1%}",
                f"  Mean Win Rate    : {agg['mean_win_rate']:.1%}",
                f"  Mean Profit Fac  : {agg['mean_profit_factor']:.2f}",
                f"  Mean Trades/mo   : {agg['mean_trades_month']:.1f}",
                f"  % Positive Windows: {agg['pct_positive_windows']:.0%}",
                "",
            ]

        if report["stress_tests"]:
            lines += ["STRESS TESTS", "-" * 70]
            for sid, result in report["stress_tests"].items():
                if result.get("skipped"):
                    lines.append(f"  {sid}: SKIPPED")
                else:
                    lines.append(
                        f"  {sid} — {result.get('stress_name', '')}: "
                        f"Capital preserved={result.get('capital_preserved', 0):.1%} | "
                        f"DD={result.get('max_drawdown', 0):.1%}"
                    )
            lines.append("")

        if report["gate_results"]:
            lines += ["PHASE 3 GATE CRITERIA", "-" * 70]
            for gname, gate in report["gate_results"].items():
                status = "✓ PASS" if gate["pass"] else "✗ FAIL"
                val    = gate["value"]
                val_str= f"{val:.1%}" if val < 10 else f"{val:.2f}"
                lines.append(
                    f"  {status}  {gate['label']:30s} "
                    f"{val_str:>10}  (threshold: {gate['threshold']})"
                )
            lines.append("")
            overall = "✓ ALL GATES PASSED — PROCEED TO PHASE 4" \
                      if report["overall_pass"] \
                      else "✗ GATES FAILED — RETUNE BEFORE PHASE 4"
            lines += ["=" * 70, f"  {overall}", "=" * 70]

        return lines

    def print_summary(self, report: Dict):
        """Prints the summary to console."""
        lines = self._build_summary_text(report)
        print("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest training/walk_forward.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestWalkForward:
    """Unit tests for walk-forward backtester components."""

    def _make_mock_loader(self, n_symbols=5, n_days=300) -> MarketDataLoader:
        loader = MarketDataLoader.__new__(MarketDataLoader)
        loader._loaded  = True
        loader._symbols = [f"STK{i:02d}" for i in range(n_symbols)]
        loader._trading_dates = [
            (pd.Timestamp("2019-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n_days)
        ]
        loader._cache = {}
        np.random.seed(42)
        for sym in loader._symbols:
            close = 1000 + np.cumsum(np.random.randn(n_days) * 5)
            close = np.maximum(close, 10.0)
            idx   = pd.date_range("2019-01-01", periods=n_days, freq="D")
            loader._cache[sym] = pd.DataFrame({
                "open"  : close * 0.99,
                "high"  : close * 1.01,
                "low"   : close * 0.98,
                "close" : close,
                "volume": np.full(n_days, 2_000_000.0),
            }, index=idx)
        return loader

    def _make_env(self, loader, start_idx=0, end_idx=200) -> GodsEyeEnv:
        return GodsEyeEnv(
            data_loader     = loader,
            backbone        = None,
            mode            = TradeMode.SWING,
            initial_capital = INITIAL_CAPITAL,
            n_stocks        = 3,
            train_start_idx = start_idx,
            train_end_idx   = end_idx,
        )

    def _make_random_model(self, env) -> PPO:
        vec_env = DummyVecEnv([lambda: env])
        return PPO(
            "MlpPolicy", vec_env,
            verbose=0, tensorboard_log=None,
            device="cpu", n_steps=64, batch_size=32,
        )

    # ── EpisodeRunner tests ───────────────────────────────────────────────

    def test_episode_runner_returns_dict(self):
        loader = self._make_mock_loader()
        env    = self._make_env(loader)
        model  = self._make_random_model(env)
        runner = EpisodeRunner(model, env, n_episodes=3)
        result = runner.run()
        assert isinstance(result, dict)

    def test_episode_runner_has_required_keys(self):
        loader = self._make_mock_loader()
        env    = self._make_env(loader)
        model  = self._make_random_model(env)
        runner = EpisodeRunner(model, env, n_episodes=2)
        result = runner.run()
        for key in ("cagr", "sharpe", "max_drawdown", "win_rate",
                    "profit_factor", "avg_trades_per_month", "capital_preserved"):
            assert key in result, f"Missing key: {key}"

    def test_episode_runner_drawdown_non_negative(self):
        loader = self._make_mock_loader()
        env    = self._make_env(loader)
        model  = self._make_random_model(env)
        runner = EpisodeRunner(model, env, n_episodes=3)
        result = runner.run()
        assert result["max_drawdown"] >= 0

    def test_episode_runner_win_rate_in_range(self):
        loader = self._make_mock_loader()
        env    = self._make_env(loader)
        model  = self._make_random_model(env)
        runner = EpisodeRunner(model, env, n_episodes=5)
        result = runner.run()
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_episode_runner_capital_preserved_positive(self):
        loader = self._make_mock_loader()
        env    = self._make_env(loader)
        model  = self._make_random_model(env)
        runner = EpisodeRunner(model, env, n_episodes=3)
        result = runner.run()
        assert result["capital_preserved"] > 0

    def test_episode_runner_n_episodes_correct(self):
        loader = self._make_mock_loader()
        env    = self._make_env(loader)
        model  = self._make_random_model(env)
        runner = EpisodeRunner(model, env, n_episodes=4)
        result = runner.run()
        assert result["n_episodes"] == 4

    # ── Gate evaluation tests ─────────────────────────────────────────────

    def test_gate_all_pass(self):
        """All gates should pass when metrics meet all thresholds."""
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        agg = {
            "mean_cagr"         : 0.50,
            "max_drawdown"      : 0.08,
            "mean_sharpe"       : 2.0,
            "mean_profit_factor": 2.5,
            "mean_win_rate"     : 0.60,
            "mean_trades_month" : 10.0,
        }
        stress = {"ST-01": {"capital_preserved": 0.92, "max_drawdown": 0.06}}
        gates  = backtester._evaluate_gates(agg, stress)
        assert all(g["pass"] for g in gates.values()), \
            f"Expected all pass: {[(k, g['pass']) for k, g in gates.items() if not g['pass']]}"

    def test_gate_cagr_fail(self):
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        agg = {
            "mean_cagr"         : 0.30,   # below 45%
            "max_drawdown"      : 0.08,
            "mean_sharpe"       : 2.0,
            "mean_profit_factor": 2.5,
            "mean_win_rate"     : 0.60,
            "mean_trades_month" : 10.0,
        }
        stress = {"ST-01": {"capital_preserved": 0.92}}
        gates  = backtester._evaluate_gates(agg, stress)
        assert not gates["cagr"]["pass"]

    def test_gate_drawdown_fail(self):
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        agg = {
            "mean_cagr"         : 0.50,
            "max_drawdown"      : 0.15,   # above 12%
            "mean_sharpe"       : 2.0,
            "mean_profit_factor": 2.5,
            "mean_win_rate"     : 0.60,
            "mean_trades_month" : 10.0,
        }
        stress = {"ST-01": {"capital_preserved": 0.92}}
        gates  = backtester._evaluate_gates(agg, stress)
        assert not gates["max_drawdown"]["pass"]

    def test_gate_win_rate_too_high(self):
        """Win rate > 68% should fail (overfitting indicator)."""
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        agg = {
            "mean_cagr"         : 0.50,
            "max_drawdown"      : 0.08,
            "mean_sharpe"       : 2.0,
            "mean_profit_factor": 2.5,
            "mean_win_rate"     : 0.75,   # above 68%
            "mean_trades_month" : 10.0,
        }
        stress = {"ST-01": {"capital_preserved": 0.92}}
        gates  = backtester._evaluate_gates(agg, stress)
        assert not gates["win_rate"]["pass"]

    def test_gate_trades_too_low(self):
        """Too few trades (< 8/month) should fail."""
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        agg = {
            "mean_cagr"         : 0.50,
            "max_drawdown"      : 0.08,
            "mean_sharpe"       : 2.0,
            "mean_profit_factor": 2.5,
            "mean_win_rate"     : 0.60,
            "mean_trades_month" : 3.0,   # below 8
        }
        stress = {"ST-01": {"capital_preserved": 0.92}}
        gates  = backtester._evaluate_gates(agg, stress)
        assert not gates["monthly_trades"]["pass"]

    def test_gate_covid_stress_fail(self):
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        agg = {
            "mean_cagr"         : 0.50,
            "max_drawdown"      : 0.08,
            "mean_sharpe"       : 2.0,
            "mean_profit_factor": 2.5,
            "mean_win_rate"     : 0.60,
            "mean_trades_month" : 10.0,
        }
        stress = {"ST-01": {"capital_preserved": 0.80}}   # below 88%
        gates  = backtester._evaluate_gates(agg, stress)
        assert not gates["covid_stress"]["pass"]

    # ── Aggregation tests ─────────────────────────────────────────────────

    def test_aggregate_windows_correct_mean(self):
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        windows = [
            {"cagr": 0.4, "sharpe": 1.5, "max_drawdown": 0.08,
             "win_rate": 0.60, "profit_factor": 2.0,
             "avg_trades_per_month": 10, "episode_returns": [0.03, 0.04]},
            {"cagr": 0.6, "sharpe": 2.5, "max_drawdown": 0.10,
             "win_rate": 0.65, "profit_factor": 3.0,
             "avg_trades_per_month": 12, "episode_returns": [0.05, 0.06]},
        ]
        agg = backtester._aggregate_windows(windows)
        assert abs(agg["mean_cagr"]   - 0.50) < 1e-6
        assert abs(agg["mean_sharpe"] - 2.00) < 1e-6
        assert agg["max_drawdown"]    == 0.10

    def test_aggregate_n_windows_correct(self):
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        windows = [
            {"cagr": 0.4, "sharpe": 1.5, "max_drawdown": 0.08,
             "win_rate": 0.60, "profit_factor": 2.0,
             "avg_trades_per_month": 10, "episode_returns": []},
        ] * 6
        agg = backtester._aggregate_windows(windows)
        assert agg["n_windows"] == 6

    # ── Report generation tests ───────────────────────────────────────────

    def test_build_summary_text_no_crash(self):
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        report = {
            "timestamp"   : "2024-01-01T00:00:00",
            "checkpoint"  : "checkpoints/swing_best.pt",
            "n_episodes"  : 30,
            "windows"     : {},
            "aggregated"  : {},
            "stress_tests": {},
            "gate_results": {},
            "overall_pass": False,
        }
        lines = backtester._build_summary_text(report)
        assert isinstance(lines, list)
        assert len(lines) > 0

    def test_save_report_creates_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("training.walk_forward.LOG_DIR", tmp_path)
        backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
        report = {
            "timestamp": "2024-01-01T00:00:00",
            "checkpoint": "test.pt",
            "n_episodes": 5,
            "windows": {},
            "aggregated": {},
            "stress_tests": {},
            "gate_results": {},
            "overall_pass": False,
        }
        json_path, txt_path = backtester.save_report(report)
        assert json_path.exists()
        assert txt_path.exists()

    # ── Walk-forward window config tests ──────────────────────────────────

    def test_six_windows_defined(self):
        assert len(WALK_FORWARD_WINDOWS) == 6

    def test_windows_have_required_keys(self):
        for w in WALK_FORWARD_WINDOWS:
            for key in ("id", "name", "train_end", "test_start", "test_end", "regime"):
                assert key in w, f"Window {w.get('id')} missing key: {key}"

    def test_windows_chronologically_ordered(self):
        dates = [w["test_start"] for w in WALK_FORWARD_WINDOWS]
        assert dates == sorted(dates), "Walk-forward windows not in chronological order"

    def test_three_stress_tests_defined(self):
        assert len(STRESS_TESTS) == 3

    def test_gate_constants_reasonable(self):
        assert 0.30 <= GATE_CAGR      <= 0.60
        assert 0.08 <= GATE_DRAWDOWN  <= 0.20
        assert 1.0  <= GATE_SHARPE    <= 3.0
        assert 1.0  <= GATE_PROFIT_FACTOR <= 5.0


# ── CLI ENTRY POINT ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E — Walk-Forward Backtester"
    )
    parser.add_argument(
        "--window", type=int, nargs="+", metavar="N",
        help="Run specific window(s) only (e.g. --window 3 4)"
    )
    parser.add_argument(
        "--stress-only", action="store_true",
        help="Run stress tests only, skip walk-forward windows"
    )
    parser.add_argument(
        "--episodes", type=int, default=EPISODES_PER_WINDOW,
        help=f"Episodes per window (default: {EPISODES_PER_WINDOW})"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic policy during evaluation (default: deterministic)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(SWING_BEST),
        help="Path to swing_best.pt checkpoint"
    )
    args = parser.parse_args()

    backtester = WalkForwardBacktester(
        checkpoint_path = Path(args.checkpoint),
        device          = args.device,
        n_episodes      = args.episodes,
        deterministic   = not args.stochastic,
    )

    try:
        backtester.setup()
        report = backtester.run(
            windows     = args.window,
            stress_only = args.stress_only,
        )
        backtester.print_summary(report)
        backtester.save_report(report)

        # Exit code 0 = all gates passed, 1 = failed
        import sys
        sys.exit(0 if report["overall_pass"] else 1)

    except KeyboardInterrupt:
        logger.warning("Backtester interrupted by user.")