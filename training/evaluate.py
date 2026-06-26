"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Phase 3 Final Evaluator                         ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : training/evaluate.py                                   ║
║         Phase   : 3 — RL Agent Training (Gate Validation)               ║
║                                                                          ║
║  What this file does:                                                    ║
║    Standalone evaluation script that loads a trained checkpoint,         ║
║    runs the complete walk-forward backtest, computes all gate metrics,   ║
║    and produces the official Phase 3 → Phase 4 go/no-go decision.       ║
║                                                                          ║
║    This is the FINAL step of Phase 3. Run it after training completes   ║
║    to determine if the model is ready for Phase 4 (paper trading).      ║
║                                                                          ║
║  Three evaluation modes:                                                 ║
║    1. FULL   : Complete 6-window walk-forward + 3 stress tests          ║
║    2. QUICK  : 2 windows + 1 stress test (fast sanity check)            ║
║    3. SINGLE : One specific window or stress test                        ║
║                                                                          ║
║  Additional analyses:                                                    ║
║    • Per-regime performance breakdown                                    ║
║    • Drawdown waterfall chart (text-based)                               ║
║    • Monthly return heatmap (text-based)                                 ║
║    • Trade distribution analysis                                         ║
║    • Risk Constitution trigger frequency report                          ║
║    • Comparison vs buy-and-hold Nifty benchmark                         ║
║                                                                          ║
║  Usage:                                                                  ║
║    # Full evaluation (recommended before Phase 4)                        ║
║    python -m training.evaluate                                           ║
║                                                                          ║
║    # Quick check (during training to monitor progress)                   ║
║    python -m training.evaluate --mode quick                              ║
║                                                                          ║
║    # Evaluate a specific checkpoint                                      ║
║    python -m training.evaluate --checkpoint checkpoints/swing_best.pt   ║
║                                                                          ║
║    # Compare two checkpoints                                             ║
║    python -m training.evaluate --compare checkpoints/swing_final.pt     ║
║                                                                          ║
║  Output:                                                                 ║
║    logs/evaluate/eval_<timestamp>.json   full results                    ║
║    logs/evaluate/eval_<timestamp>.txt    human-readable report           ║
║    logs/evaluate/eval_latest.txt         always overwritten (latest)     ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install stable-baselines3 torch numpy pandas psycopg2-binary     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch

from datetime  import datetime
from pathlib   import Path
from typing    import Dict, List, Optional, Tuple
from loguru    import logger
from dotenv    import load_dotenv

from stable_baselines3            import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.godseye_env      import GodsEyeEnv, MarketDataLoader, TradeMode
from environment.risk_constitution import RiskConstitution, PortfolioState, MarketState
from models.backbone              import GodsEyeBackbone
from training.walk_forward        import (
    WalkForwardBacktester,
    EpisodeRunner,
    WALK_FORWARD_WINDOWS,
    STRESS_TESTS,
    GATE_CAGR, GATE_DRAWDOWN, GATE_SHARPE,
    GATE_PROFIT_FACTOR, GATE_WIN_RATE_MIN, GATE_WIN_RATE_MAX,
    GATE_TRADES_MIN, GATE_TRADES_MAX,
    INITIAL_CAPITAL, EPISODES_PER_WINDOW, N_STOCKS,
)

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR        = ROOT_DIR / "logs" / "evaluate"
SWING_BEST     = CHECKPOINT_DIR / "swing_best.pt"
SWING_FINAL    = CHECKPOINT_DIR / "swing_final.pt"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Quick mode windows (subset for fast evaluation) ───────────────────────
QUICK_WINDOWS = [3, 5]   # bear + recovery — most informative pair


# ══════════════════════════════════════════════════════════════════════════
#  BENCHMARK (Buy-and-Hold Nifty comparison)
# ══════════════════════════════════════════════════════════════════════════

class NiftyBenchmark:
    """
    Computes buy-and-hold Nifty 50 returns for the same test periods.
    Used to contextualise the model's performance.

    Uses NIFTY BEES (NSE: NIFTYBEES) as a proxy — it's in daily_ohlcv.
    Falls back to equal-weight of top-20 liquid stocks if NIFTYBEES absent.
    """

    NIFTY_PROXY = "NIFTYBEES"   # ETF tracking Nifty 50

    def __init__(self, data_loader: MarketDataLoader):
        self.data_loader = data_loader

    def get_return(self, start_date: str, end_date: str) -> Optional[float]:
        """
        Returns buy-and-hold return for Nifty proxy over the date range.

        Args:
            start_date : YYYY-MM-DD
            end_date   : YYYY-MM-DD

        Returns:
            fractional return (e.g. 0.15 = 15%) or None if data unavailable
        """
        # Try NIFTYBEES first
        df = self.data_loader.get_ohlcv(self.NIFTY_PROXY)
        if df.empty:
            return self._fallback_return(start_date, end_date)

        mask = (df.index >= pd.Timestamp(start_date)) & \
               (df.index <= pd.Timestamp(end_date))
        period = df[mask]

        if len(period) < 2:
            return None

        entry = float(period.iloc[0]["close"])
        exit_ = float(period.iloc[-1]["close"])
        return (exit_ - entry) / entry

    def _fallback_return(self, start_date: str, end_date: str) -> Optional[float]:
        """Equal-weight return across all available symbols as Nifty proxy."""
        returns = []
        for sym in self.data_loader.symbols[:50]:   # use top 50 symbols
            df = self.data_loader.get_ohlcv(sym)
            if df.empty:
                continue
            mask = (df.index >= pd.Timestamp(start_date)) & \
                   (df.index <= pd.Timestamp(end_date))
            period = df[mask]
            if len(period) < 2:
                continue
            ret = (float(period.iloc[-1]["close"]) -
                   float(period.iloc[0]["close"])) / float(period.iloc[0]["close"])
            returns.append(ret)
        return float(np.mean(returns)) if returns else None


# ══════════════════════════════════════════════════════════════════════════
#  DETAILED EPISODE ANALYSER
# ══════════════════════════════════════════════════════════════════════════

class DetailedAnalyser:
    """
    Runs extended analysis on top of EpisodeRunner output.

    Produces:
        - Monthly return distribution
        - Drawdown waterfall (text chart)
        - Trade frequency distribution
        - RC trigger frequency
        - Win/loss streak analysis
    """

    def __init__(self, model: PPO, env: GodsEyeEnv, n_episodes: int = 50):
        self.model      = model
        self.env        = env
        self.n_episodes = n_episodes

    def run_extended(self) -> Dict:
        """
        Runs extended analysis episodes and returns detailed statistics.
        """
        all_returns     : List[float] = []
        all_drawdowns   : List[float] = []
        all_trades      : List[int]   = []
        win_loss_sequence: List[bool] = []
        rc_triggers     : Dict[str, int] = {}

        for ep in range(self.n_episodes):
            obs, _ = self.env.reset(seed=ep * 31 + 7)
            done   = False
            pv_hist= [INITIAL_CAPITAL]
            peak   = INITIAL_CAPITAL
            max_dd = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.env.step(int(action))
                done = terminated or truncated

                pv = info.get("portfolio_value", INITIAL_CAPITAL)
                pv_hist.append(pv)

                if pv > peak:
                    peak = pv
                dd = (peak - pv) / peak
                max_dd = max(max_dd, dd)

                # Collect RC trigger data
                for rule, count in info.get("rc_triggered", {}).items():
                    rc_triggers[rule] = rc_triggers.get(rule, 0) + count

            final_ret = (pv_hist[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
            all_returns.append(final_ret)
            all_drawdowns.append(max_dd)
            all_trades.append(info.get("total_trades", 0))
            win_loss_sequence.append(final_ret > 0)

        return {
            "return_distribution"  : self._distribution_stats(all_returns),
            "drawdown_distribution": self._distribution_stats(all_drawdowns),
            "trade_distribution"   : self._distribution_stats(all_trades),
            "win_loss_streaks"     : self._streak_analysis(win_loss_sequence),
            "rc_trigger_frequency" : rc_triggers,
            "monthly_returns"      : self._monthly_return_chart(all_returns),
            "drawdown_waterfall"   : self._drawdown_waterfall(all_drawdowns),
        }

    def _distribution_stats(self, values: List) -> Dict:
        """Returns distribution statistics for a list of values."""
        if not values:
            return {}
        arr = np.array(values)
        return {
            "mean"  : float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std"   : float(np.std(arr)),
            "min"   : float(np.min(arr)),
            "max"   : float(np.max(arr)),
            "p25"   : float(np.percentile(arr, 25)),
            "p75"   : float(np.percentile(arr, 75)),
            "p5"    : float(np.percentile(arr, 5)),
            "p95"   : float(np.percentile(arr, 95)),
        }

    def _streak_analysis(self, sequence: List[bool]) -> Dict:
        """Analyses win/loss streaks."""
        if not sequence:
            return {}

        max_win_streak  = 0
        max_loss_streak = 0
        cur_win         = 0
        cur_loss        = 0

        for is_win in sequence:
            if is_win:
                cur_win  += 1
                cur_loss  = 0
                max_win_streak = max(max_win_streak, cur_win)
            else:
                cur_loss += 1
                cur_win   = 0
                max_loss_streak = max(max_loss_streak, cur_loss)

        return {
            "max_win_streak"   : max_win_streak,
            "max_loss_streak"  : max_loss_streak,
            "total_episodes"   : len(sequence),
            "total_wins"       : sum(sequence),
            "total_losses"     : len(sequence) - sum(sequence),
        }

    def _monthly_return_chart(self, returns: List[float]) -> str:
        """
        Builds a simple text-based monthly return distribution bar chart.
        Each bar represents one return bucket.
        """
        if not returns:
            return ""

        arr      = np.array(returns)
        buckets  = np.linspace(arr.min(), arr.max(), 11)
        counts,_ = np.histogram(arr, bins=buckets)
        max_count= max(counts) if counts.max() > 0 else 1

        lines = ["  Return Distribution (episode returns):", ""]
        for i, (low, high) in enumerate(zip(buckets[:-1], buckets[1:])):
            bar_len  = int(counts[i] / max_count * 30)
            bar      = "█" * bar_len
            pct_line = f"  {low:+.1%} to {high:+.1%} | {bar} ({counts[i]})"
            lines.append(pct_line)
        return "\n".join(lines)

    def _drawdown_waterfall(self, drawdowns: List[float]) -> str:
        """Builds a text-based drawdown severity chart."""
        if not drawdowns:
            return ""

        buckets = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 1.0]
        labels  = ["0–2%", "2–5%", "5–8%", "8–10%", "10–12%", ">12%"]
        arr     = np.array(drawdowns)
        counts  = [
            int(((arr >= lo) & (arr < hi)).sum())
            for lo, hi in zip(buckets[:-1], buckets[1:])
        ]
        total    = len(drawdowns)
        lines    = ["  Drawdown Severity Distribution:", ""]
        for label, count in zip(labels, counts):
            bar_len = int(count / total * 30) if total > 0 else 0
            bar     = "█" * bar_len
            pct     = count / total if total > 0 else 0
            lines.append(f"  {label:8s} | {bar} {pct:.0%} ({count} episodes)")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATOR
# ══════════════════════════════════════════════════════════════════════════

class PhaseEvaluator:
    """
    Orchestrates the complete Phase 3 gate evaluation.

    Wraps WalkForwardBacktester with additional analyses:
        - Benchmark comparison
        - Detailed distribution analysis
        - RC trigger frequency
        - Formatted console + file report

    Usage:
        evaluator = PhaseEvaluator()
        evaluator.setup()
        report = evaluator.evaluate(mode="full")
        # Exit code 0 = pass, 1 = fail
    """

    def __init__(
        self,
        checkpoint_path  : Path = SWING_BEST,
        compare_path     : Optional[Path] = None,
        device           : str  = "cuda" if torch.cuda.is_available() else "cpu",
        n_episodes       : int  = EPISODES_PER_WINDOW,
        deterministic    : bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.compare_path    = compare_path
        self.device          = device
        self.n_episodes      = n_episodes
        self.deterministic   = deterministic

        self.backtester : Optional[WalkForwardBacktester] = None
        self.benchmark  : Optional[NiftyBenchmark]        = None

    def setup(self):
        """Initialises backtester and benchmark."""
        logger.info("PhaseEvaluator: initialising...")
        self.backtester = WalkForwardBacktester(
            checkpoint_path = self.checkpoint_path,
            device          = self.device,
            n_episodes      = self.n_episodes,
            deterministic   = self.deterministic,
        )
        self.backtester.setup()
        self.benchmark = NiftyBenchmark(self.backtester.data_loader)
        logger.info("PhaseEvaluator ready.")

    def evaluate(self, mode: str = "full") -> Dict:
        """
        Runs the evaluation in the specified mode.

        Args:
            mode : 'full', 'quick', or 'single'

        Returns:
            Complete evaluation report dict
        """
        logger.info(f"Starting Phase 3 evaluation — mode={mode.upper()}")
        start_time = time.time()

        # ── Select windows based on mode ──────────────────────────────────
        if mode == "quick":
            windows     = QUICK_WINDOWS
            n_eps       = min(self.n_episodes, 10)
            stress_only = False
        elif mode == "stress":
            windows     = None
            n_eps       = self.n_episodes
            stress_only = True
        else:  # full
            windows     = None
            n_eps       = self.n_episodes
            stress_only = False

        self.backtester.n_episodes = n_eps

        # ── Run walk-forward backtest ──────────────────────────────────────
        wf_report = self.backtester.run(
            windows     = windows,
            stress_only = stress_only,
        )

        # ── Add benchmark comparison ───────────────────────────────────────
        benchmark_returns = {}
        for window in WALK_FORWARD_WINDOWS:
            if windows and window["id"] not in windows:
                continue
            ret = self.benchmark.get_return(
                window["test_start"], window["test_end"]
            )
            benchmark_returns[window["id"]] = ret

        wf_report["benchmark_returns"] = benchmark_returns
        wf_report["alpha"] = self._compute_alpha(
            wf_report.get("windows", {}),
            benchmark_returns,
        )

        # ── Run detailed analysis on val period ───────────────────────────
        if mode == "full" and wf_report.get("windows"):
            logger.info("Running detailed distribution analysis...")
            try:
                # Use window 6 (most recent) for detailed analysis
                w6_config = next(
                    w for w in WALK_FORWARD_WINDOWS if w["id"] == 6
                )
                detail_env = self.backtester._make_env(
                    w6_config["test_start"], w6_config["test_end"]
                )
                analyser = DetailedAnalyser(
                    self.backtester.model, detail_env,
                    n_episodes=min(n_eps, 20),
                )
                wf_report["detailed_analysis"] = analyser.run_extended()
            except Exception as e:
                logger.warning(f"Detailed analysis failed: {e}")
                wf_report["detailed_analysis"] = {}

        # ── Compare against second checkpoint if provided ─────────────────
        if self.compare_path and self.compare_path.exists():
            logger.info(f"Comparing against {self.compare_path}...")
            wf_report["comparison"] = self._compare_checkpoint(
                windows, stress_only, n_eps
            )

        elapsed = time.time() - start_time
        wf_report["eval_elapsed_s"] = elapsed
        wf_report["eval_mode"]      = mode

        return wf_report

    def _compute_alpha(
        self,
        windows          : Dict,
        benchmark_returns : Dict,
    ) -> Dict:
        """
        Computes alpha (model return - benchmark return) per window.
        Positive alpha = model outperforms buy-and-hold Nifty.
        """
        alphas = {}
        for wid_str, result in windows.items():
            wid = int(wid_str)
            bench = benchmark_returns.get(wid)
            if bench is not None:
                model_ret = result.get("avg_episode_return", 0.0)
                alphas[wid] = {
                    "model_return"    : model_ret,
                    "benchmark_return": bench,
                    "alpha"           : model_ret - bench,
                }
        return alphas

    def _compare_checkpoint(
        self,
        windows    : Optional[List[int]],
        stress_only: bool,
        n_episodes : int,
    ) -> Dict:
        """Runs the same evaluation on the comparison checkpoint."""
        compare_bt = WalkForwardBacktester(
            checkpoint_path = self.compare_path,
            device          = self.device,
            n_episodes      = n_episodes,
            deterministic   = self.deterministic,
        )
        compare_bt.data_loader = self.backtester.data_loader  # reuse data
        compare_bt.backbone    = self.backtester.backbone      # reuse backbone

        # Rebuild model with comparison checkpoint
        dummy_env = self.backtester._make_env("2021-01-01", "2021-06-30")
        vec_env   = DummyVecEnv([lambda: dummy_env])
        compare_bt.model = PPO(
            "MlpPolicy", vec_env,
            device=self.device, verbose=0, tensorboard_log=None,
        )
        ckpt = torch.load(
            self.compare_path, map_location=self.device, weights_only=False
        )
        if "ppo_policy_state" in ckpt:
            compare_bt.model.policy.load_state_dict(
                ckpt["ppo_policy_state"], strict=False
            )

        return compare_bt.run(windows=windows, stress_only=stress_only)

    def print_and_save(self, report: Dict) -> Tuple[Path, Path]:
        """
        Prints formatted report to console and saves to files.

        Returns:
            (json_path, txt_path)
        """
        # ── Console output ────────────────────────────────────────────────
        lines = self._build_full_report(report)
        print("\n".join(lines))

        # ── Save files ────────────────────────────────────────────────────
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path= LOG_DIR / f"eval_{ts}.json"
        txt_path = LOG_DIR / f"eval_{ts}.txt"
        latest   = LOG_DIR / "eval_latest.txt"

        def convert(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.ndarray,)):  return obj.tolist()
            if isinstance(obj, Path):           return str(obj)
            return obj

        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=convert)

        content = "\n".join(lines)
        with open(txt_path,  "w", encoding="utf-8") as f:
            f.write(content)
        with open(latest,    "w", encoding="utf-8") as f:
            f.write(content)

        logger.success(f"Evaluation saved to {txt_path}")
        return json_path, txt_path

    def _build_full_report(self, report: Dict) -> List[str]:
        """Builds the complete human-readable evaluation report."""
        lines = []
        sep   = "═" * 72

        # ── Header ────────────────────────────────────────────────────────
        lines += [
            sep,
            "  G.O.D.S E.Y.E — Phase 3 Evaluation Report",
            f"  Checkpoint : {report.get('checkpoint', 'unknown')}",
            f"  Mode       : {report.get('eval_mode', 'full').upper()}",
            f"  Timestamp  : {report.get('timestamp', '')}",
            f"  Elapsed    : {report.get('eval_elapsed_s', 0)/60:.1f} minutes",
            sep,
            "",
        ]

        # ── Per-window results ────────────────────────────────────────────
        if report.get("windows"):
            lines += ["  WALK-FORWARD WINDOW RESULTS", "  " + "─" * 68]
            alpha_data = report.get("alpha", {})

            for wid, result in report["windows"].items():
                wid_int  = int(wid)
                config   = next(
                    (w for w in WALK_FORWARD_WINDOWS if w["id"] == wid_int),
                    {}
                )
                alpha    = alpha_data.get(wid_int, {})
                cagr     = result["cagr"]
                status   = "✓" if cagr >= 0 else "✗"

                lines += [
                    f"  {status} Window {wid} — {result.get('window_name', '')}",
                    f"    Period  : {result.get('test_start','')} → {result.get('test_end','')}",
                    f"    Regime  : {config.get('regime', '')}",
                    f"    CAGR    : {cagr:.1%}"
                    + (f"  (Nifty: {alpha.get('benchmark_return',0):.1%}  "
                       f"Alpha: {alpha.get('alpha',0):+.1%})"
                       if alpha else ""),
                    f"    Sharpe  : {result['sharpe']:.3f}",
                    f"    Max DD  : {result['max_drawdown']:.1%}",
                    f"    Win Rate: {result['win_rate']:.1%}",
                    f"    PF      : {result['profit_factor']:.2f}",
                    f"    Trades  : {result['avg_trades_per_month']:.1f}/month",
                    "",
                ]

        # ── Aggregated ────────────────────────────────────────────────────
        if report.get("aggregated"):
            agg = report["aggregated"]
            lines += [
                "  AGGREGATED METRICS (ALL WINDOWS)",
                "  " + "─" * 68,
                f"    Mean CAGR          : {agg['mean_cagr']:.1%}",
                f"    Median CAGR        : {agg['median_cagr']:.1%}",
                f"    Min CAGR (worst)   : {agg['min_cagr']:.1%}",
                f"    Mean Sharpe        : {agg['mean_sharpe']:.3f}",
                f"    Max Drawdown       : {agg['max_drawdown']:.1%}",
                f"    Mean Win Rate      : {agg['mean_win_rate']:.1%}",
                f"    Mean Profit Factor : {agg['mean_profit_factor']:.2f}",
                f"    Mean Trades/month  : {agg['mean_trades_month']:.1f}",
                f"    % Positive Windows : {agg['pct_positive_windows']:.0%}",
                "",
            ]

        # ── Stress tests ──────────────────────────────────────────────────
        if report.get("stress_tests"):
            lines += ["  STRESS TEST RESULTS", "  " + "─" * 68]
            for sid, result in report["stress_tests"].items():
                if result.get("skipped"):
                    lines.append(f"    {sid}: SKIPPED — {result.get('reason','')}")
                else:
                    preserved = result.get("capital_preserved", 1.0)
                    dd        = result.get("max_drawdown", 0.0)
                    status    = "✓" if preserved >= 0.88 else "✗"
                    lines.append(
                        f"    {status} {sid} — {result.get('stress_name','')}: "
                        f"Capital={preserved:.1%}  DD={dd:.1%}"
                    )
            lines.append("")

        # ── Detailed analysis ─────────────────────────────────────────────
        da = report.get("detailed_analysis", {})
        if da:
            lines += ["  DETAILED ANALYSIS (Window 6 — Most Recent)", "  " + "─" * 68]

            streaks = da.get("win_loss_streaks", {})
            if streaks:
                lines += [
                    f"    Max win streak   : {streaks.get('max_win_streak', 0)} episodes",
                    f"    Max loss streak  : {streaks.get('max_loss_streak', 0)} episodes",
                ]

            rc = da.get("rc_trigger_frequency", {})
            if rc:
                lines.append("    RC triggers (total across episodes):")
                for rule, count in sorted(rc.items()):
                    lines.append(f"      {rule}: {count} times")

            monthly = da.get("monthly_returns", "")
            if monthly:
                lines += ["", monthly]

            dd_chart = da.get("drawdown_waterfall", "")
            if dd_chart:
                lines += ["", dd_chart]

            lines.append("")

        # ── Comparison ────────────────────────────────────────────────────
        if report.get("comparison") and report["comparison"].get("aggregated"):
            comp_agg = report["comparison"]["aggregated"]
            main_agg = report.get("aggregated", {})
            lines += [
                "  CHECKPOINT COMPARISON",
                "  " + "─" * 68,
                f"    {'Metric':25s}  {'Primary':>10}  {'Compare':>10}  {'Delta':>10}",
                f"    {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}",
            ]
            metrics = [
                ("Mean CAGR",     "mean_cagr",          ".1%"),
                ("Mean Sharpe",   "mean_sharpe",         ".3f"),
                ("Max Drawdown",  "max_drawdown",        ".1%"),
                ("Win Rate",      "mean_win_rate",       ".1%"),
                ("Profit Factor", "mean_profit_factor",  ".2f"),
            ]
            for label, key, fmt in metrics:
                main_val = main_agg.get(key, 0)
                comp_val = comp_agg.get(key, 0)
                delta    = main_val - comp_val
                lines.append(
                    f"    {label:25s}  "
                    f"{format(main_val, fmt):>10}  "
                    f"{format(comp_val, fmt):>10}  "
                    f"{delta:+.3f}"
                )
            lines.append("")

        # ── Gate results ──────────────────────────────────────────────────
        if report.get("gate_results"):
            lines += ["  PHASE 3 GATE CRITERIA", "  " + "─" * 68]
            for gname, gate in report["gate_results"].items():
                status  = "✓ PASS" if gate["pass"] else "✗ FAIL"
                val     = gate["value"]
                val_str = f"{val:.1%}" if isinstance(val, float) and val < 10 \
                          else f"{val:.2f}"
                lines.append(
                    f"    {status}  {gate['label']:32s} "
                    f"{val_str:>10}   threshold: {gate['threshold']}"
                )
            lines.append("")

            overall = report.get("overall_pass", False)
            decision_line = (
                "✓  ALL GATES PASSED — PROCEED TO PHASE 4 (PAPER TRADING)"
                if overall else
                "✗  GATES FAILED — DO NOT PROCEED — RETUNE AND RETRAIN"
            )
            lines += [
                sep,
                f"  {decision_line}",
                sep,
            ]

            if not overall:
                lines += [
                    "",
                    "  RECOMMENDATIONS FOR FAILED GATES:",
                    "  " + "─" * 68,
                ]
                failed = [
                    (k, g) for k, g in report["gate_results"].items()
                    if not g["pass"]
                ]
                for gname, gate in failed:
                    lines.append(
                        f"  ✗ {gate['label']}: "
                        f"{gate['value']:.3f} vs threshold {gate['threshold']}"
                    )
                    advice = self._get_tuning_advice(gname, gate["value"])
                    if advice:
                        lines.append(f"    → {advice}")
                lines.append("")

        return lines

    def _get_tuning_advice(self, gate_name: str, value: float) -> str:
        """Returns specific tuning advice for each failed gate."""
        advice = {
            "cagr": (
                "Increase ENT_COEF to force more exploration. "
                "Reduce N_STOCKS to shrink obs space. "
                "Train for more steps (current: insufficient)."
            ),
            "max_drawdown": (
                "Increase drawdown penalty weight in reward_fn.py (currently 2.0). "
                "Lower RC-01 threshold from 12% to 10% in risk_constitution.py."
            ),
            "sharpe": (
                "Increase Sharpe bonus weight in reward_fn.py (currently 0.30). "
                "Reduce overtrading penalty to allow more selective trades."
            ),
            "profit_factor": (
                "Tighten SL from 1.5% to 1.0% to cut losers faster. "
                "Increase TP from 4% to 5% to let winners run longer."
            ),
            "win_rate": (
                "Win rate outside 55–68% range. "
                "If < 55%: increase confidence threshold in position_sizer.py. "
                "If > 68%: reduce confidence threshold — may be overfitting."
            ),
            "monthly_trades": (
                "Trade count outside 8–15/month range. "
                "If < 8: reduce overtrading penalty to encourage more activity. "
                "If > 15: increase overtrading penalty in reward_fn.py."
            ),
            "covid_stress": (
                "RC-06 and RC-09 must fire correctly during panic events. "
                "Check that MDS score updates daily and feeds into position sizing."
            ),
        }
        return advice.get(gate_name, "")


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest training/evaluate.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestEvaluate:
    """Unit tests for evaluation components."""

    def _make_mock_loader(self, n_symbols=5, n_days=400) -> MarketDataLoader:
        loader = MarketDataLoader.__new__(MarketDataLoader)
        loader._loaded  = True
        loader._symbols = [f"STK{i:02d}" for i in range(n_symbols)]
        loader._trading_dates = [
            (pd.Timestamp("2019-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n_days)
        ]
        loader._cache = {}
        np.random.seed(0)
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

    # ── NiftyBenchmark tests ──────────────────────────────────────────────

    def test_benchmark_returns_float_or_none(self):
        loader = self._make_mock_loader()
        bench  = NiftyBenchmark(loader)
        result = bench.get_return("2019-01-01", "2019-06-30")
        assert result is None or isinstance(result, float)

    def test_benchmark_fallback_returns_float(self):
        loader = self._make_mock_loader()
        bench  = NiftyBenchmark(loader)
        result = bench._fallback_return("2019-01-01", "2019-06-30")
        assert result is None or isinstance(result, float)

    def test_benchmark_empty_range(self):
        loader = self._make_mock_loader()
        bench  = NiftyBenchmark(loader)
        # Date range with no data
        result = bench.get_return("2030-01-01", "2030-06-30")
        assert result is None

    # ── DetailedAnalyser tests ────────────────────────────────────────────

    def _make_env_and_model(self, loader):
        from environment.godseye_env import GodsEyeEnv
        env = GodsEyeEnv(
            data_loader=loader, backbone=None,
            mode=TradeMode.SWING, n_stocks=3,
            train_start_idx=0, train_end_idx=200,
        )
        vec = DummyVecEnv([lambda: env])
        model = PPO(
            "MlpPolicy", vec, verbose=0,
            tensorboard_log=None, device="cpu",
            n_steps=64, batch_size=32,
        )
        return env, model

    def test_detailed_analyser_returns_dict(self):
        loader    = self._make_mock_loader()
        env, model= self._make_env_and_model(loader)
        analyser  = DetailedAnalyser(model, env, n_episodes=2)
        result    = analyser.run_extended()
        assert isinstance(result, dict)

    def test_detailed_analyser_has_required_keys(self):
        loader    = self._make_mock_loader()
        env, model= self._make_env_and_model(loader)
        analyser  = DetailedAnalyser(model, env, n_episodes=2)
        result    = analyser.run_extended()
        for key in ("return_distribution", "drawdown_distribution",
                    "win_loss_streaks", "rc_trigger_frequency"):
            assert key in result, f"Missing key: {key}"

    def test_distribution_stats_correct(self):
        analyser = DetailedAnalyser.__new__(DetailedAnalyser)
        values   = [0.1, 0.2, 0.3, 0.4, 0.5]
        stats    = analyser._distribution_stats(values)
        assert abs(stats["mean"] - 0.3) < 1e-6
        assert abs(stats["min"]  - 0.1) < 1e-6
        assert abs(stats["max"]  - 0.5) < 1e-6

    def test_streak_analysis_correct(self):
        analyser = DetailedAnalyser.__new__(DetailedAnalyser)
        sequence = [True, True, True, False, False, True]
        streaks  = analyser._streak_analysis(sequence)
        assert streaks["max_win_streak"]  == 3
        assert streaks["max_loss_streak"] == 2
        assert streaks["total_wins"]      == 4

    def test_streak_analysis_empty(self):
        analyser = DetailedAnalyser.__new__(DetailedAnalyser)
        result   = analyser._streak_analysis([])
        assert result == {}

    def test_monthly_return_chart_no_crash(self):
        analyser = DetailedAnalyser.__new__(DetailedAnalyser)
        chart    = analyser._monthly_return_chart([0.1, -0.05, 0.08, -0.02])
        assert isinstance(chart, str)

    def test_drawdown_waterfall_no_crash(self):
        analyser = DetailedAnalyser.__new__(DetailedAnalyser)
        chart    = analyser._drawdown_waterfall([0.01, 0.05, 0.08, 0.12])
        assert isinstance(chart, str)
        assert "2–5%" in chart

    # ── PhaseEvaluator report tests ───────────────────────────────────────

    def test_build_full_report_no_crash(self):
        evaluator = PhaseEvaluator.__new__(PhaseEvaluator)
        report = {
            "timestamp"        : "2024-01-01T00:00:00",
            "checkpoint"       : "test.pt",
            "eval_mode"        : "full",
            "eval_elapsed_s"   : 120.0,
            "windows"          : {},
            "aggregated"       : {},
            "stress_tests"     : {},
            "gate_results"     : {},
            "overall_pass"     : False,
            "benchmark_returns": {},
            "alpha"            : {},
            "detailed_analysis": {},
        }
        lines = evaluator._build_full_report(report)
        assert isinstance(lines, list)
        assert len(lines) > 5

    def test_build_report_with_gate_results(self):
        evaluator = PhaseEvaluator.__new__(PhaseEvaluator)
        report = {
            "timestamp": "2024-01-01", "checkpoint": "test.pt",
            "eval_mode": "full", "eval_elapsed_s": 60,
            "windows": {}, "aggregated": {}, "stress_tests": {},
            "benchmark_returns": {}, "alpha": {}, "detailed_analysis": {},
            "gate_results": {
                "cagr": {"pass": True,  "value": 0.50, "threshold": ">= 45%", "label": "CAGR"},
                "sharpe": {"pass": False, "value": 1.2,  "threshold": ">= 1.8", "label": "Sharpe"},
            },
            "overall_pass": False,
        }
        lines = evaluator._build_full_report(report)
        full  = "\n".join(lines)
        assert "✓ PASS" in full
        assert "✗ FAIL" in full
        assert "GATES FAILED" in full

    def test_get_tuning_advice_known_gates(self):
        evaluator = PhaseEvaluator.__new__(PhaseEvaluator)
        for gate in ("cagr", "max_drawdown", "sharpe", "profit_factor",
                     "win_rate", "monthly_trades", "covid_stress"):
            advice = evaluator._get_tuning_advice(gate, 0.5)
            assert isinstance(advice, str)

    def test_get_tuning_advice_unknown_gate(self):
        evaluator = PhaseEvaluator.__new__(PhaseEvaluator)
        advice    = evaluator._get_tuning_advice("nonexistent_gate", 0.5)
        assert advice == ""

    def test_compute_alpha_correct(self):
        evaluator = PhaseEvaluator.__new__(PhaseEvaluator)
        windows   = {
            1: {"avg_episode_return": 0.10},
            2: {"avg_episode_return": 0.05},
        }
        benchmark = {1: 0.06, 2: 0.08}
        alpha     = evaluator._compute_alpha(windows, benchmark)
        assert abs(alpha[1]["alpha"] - 0.04) < 1e-6
        assert abs(alpha[2]["alpha"] - (-0.03)) < 1e-6

    def test_save_report_creates_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("training.evaluate.LOG_DIR", tmp_path)
        evaluator = PhaseEvaluator.__new__(PhaseEvaluator)
        report = {
            "timestamp": "2024-01-01", "checkpoint": "test.pt",
            "eval_mode": "quick", "eval_elapsed_s": 10,
            "windows": {}, "aggregated": {}, "stress_tests": {},
            "gate_results": {}, "overall_pass": False,
            "benchmark_returns": {}, "alpha": {}, "detailed_analysis": {},
        }
        json_path, txt_path = evaluator.print_and_save(report)
        assert json_path.exists()
        assert txt_path.exists()
        latest = tmp_path / "eval_latest.txt"
        assert latest.exists()


# ── CLI ENTRY POINT ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E — Phase 3 Final Evaluator"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "stress"],
        default="full",
        help=(
            "full   : Complete 6-window walk-forward + 3 stress tests (default)\n"
            "quick  : 2 windows + fast sanity check\n"
            "stress : Stress tests only"
        )
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(SWING_BEST),
        help="Path to checkpoint (default: checkpoints/swing_best.pt)"
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Path to second checkpoint for comparison"
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
        help="Use stochastic policy (default: deterministic)"
    )
    args = parser.parse_args()

    evaluator = PhaseEvaluator(
        checkpoint_path = Path(args.checkpoint),
        compare_path    = Path(args.compare) if args.compare else None,
        device          = args.device,
        n_episodes      = args.episodes,
        deterministic   = not args.stochastic,
    )

    try:
        evaluator.setup()
        report = evaluator.evaluate(mode=args.mode)
        _, txt_path = evaluator.print_and_save(report)
        logger.info(f"Full report: {txt_path}")
        sys.exit(0 if report.get("overall_pass") else 1)

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted.")
        sys.exit(1)