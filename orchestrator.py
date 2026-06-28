"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Daily Trading Orchestrator                      ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : orchestrator.py  (place in project root)              ║
║                                                                          ║
║  Single file that runs everything automatically.                         ║
║  Leave running all day. It knows what to do at each time.               ║
║                                                                          ║
║  Daily schedule:                                                         ║
║    08:00 AM  Auth check (refresh Kite token)                            ║
║    08:05 AM  Morning check (black swan monitor on open positions)       ║
║    09:15 AM  Market opens — monitor positions every 60 sec              ║
║    03:30 PM  Market closes — stop monitoring                            ║
║    06:00 PM  Evening pipeline:                                          ║
║                Download official close (bhavcopy)                       ║
║                Download FII/DII final                                   ║
║                Recompute features with official close                   ║
║                Recompute embeddings from features_fused (all 28 feats) ║
║    06:45 PM  Paper trading step (ONE decision, at official close price) ║
║               → Exactly replicates backtest behavior                   ║
║               → Entry price = today's official close                   ║
║               → TP/SL = dynamic ATR-based (same as training)           ║
║               → Position sizing = fractional Kelly (same as training)  ║
║    06:50 PM  Daily report                                               ║
║                                                                          ║
║  Run:                                                                    ║
║    python orchestrator.py                                                ║
║                                                                          ║
║  Manual override:                                                        ║
║    python orchestrator.py --run-now status                              ║
║    python orchestrator.py --run-now auth                                ║
║    python orchestrator.py --run-now evening                             ║
║    python orchestrator.py --run-now trade                               ║
║    python orchestrator.py --run-now report                              ║
║    python orchestrator.py --run-now backfill --days 5                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import subprocess
import numpy as np
import psycopg2
import psycopg2.extras
import torch

from datetime  import datetime, date, timedelta
from pathlib   import Path
from loguru    import logger
from dotenv    import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent
STATE_FILE      = ROOT / ".orchestrator_state.json"
CHECKPOINT_PATH = ROOT / "checkpoints" / "swing_best.pt"

VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = ROOT / ".venv" / "bin" / "python"
if not VENV_PY.exists():
    VENV_PY = Path(sys.executable)

# ── Config ─────────────────────────────────────────────────────────────────
DB_URL          = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000000"))

# ── Trading constants (must match training exactly) ────────────────────────
N_STOCKS        = 46
EMBEDDING_DIM   = 128
PORTFOLIO_DIM   = 8
OBS_DIM         = N_STOCKS * EMBEDDING_DIM + PORTFOLIO_DIM  # 5896
SEQ_LEN         = 60
MIN_CONFIDENCE  = 0.55
MAX_POSITIONS   = 4
MAX_TRADES_MONTH= 15
FEATURE_LOOKBACK= 130   # days to recompute features

# ── Schedule (24h IST) ─────────────────────────────────────────────────────
SCHEDULE = {
    "auth"          : (8,  0),
    "morning_check" : (8,  5),
    "market_open"   : (9, 15),
    "market_close"  : (15, 31),
    "evening_start" : (18,  0),
    "trade_step"    : (18, 45),
    "daily_report"  : (18, 50),
    "sleep_until"   : (8,  0),   # next day
}

MONITOR_INTERVAL = 60   # seconds between position checks during market hours
AX_CYCLE_WARN_SEC  = 120   # log a warning if a single cycle exceeds this

ACTIONS = {0: "HOLD", 1: "BUY", 2: "STRONG_BUY", 3: "SELL", 4: "STRONG_SELL"}

FEATURE_COLS = [
    "f00_trend_score",        "f01_ema_ribbon_gap",
    "f02_adx_normalized",     "f03_supertrend_dir",
    "f04_price_vs_ema200",    "f05_swing_structure",
    "f06_msi_signal",         "f07_vrsi_normalized",
    "f08_mfi_normalized",     "f09_msi_divergence",
    "f10_mds_continuous",     "f11_fii_norm",
    "f12_dii_norm",           "f13_sentiment_score",
    "f14_sentiment_momentum", "f15_event_flag",
    "f16_market_fear_greed_n","f17_volatility_score",
    "f18_atr_pct_normalized", "f19_vol_regime_code_n",
    "f20_hv_percentile_n",    "f21_correlation_score",
    "f22_sector_divergence_n","f23_lead_lag_score",
    "f24_peer_corr_mean",     "f25_delivery_mom_n",
    "f26_swing_tp_normalized","f27_swing_sl_normalized",
]


# ══════════════════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        return s if s.get("date") == str(date.today()) else {}
    except Exception:
        return {}


def save_state(state: dict):
    state["date"] = str(date.today())
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def mark_done(state: dict, task: str):
    state[task] = {"done": True, "time": datetime.now().strftime("%H:%M:%S")}
    save_state(state)
    logger.success(f"✓ {task} done at {state[task]['time']}")


def is_done(state: dict, task: str) -> bool:
    return state.get(task, {}).get("done", False)


# ══════════════════════════════════════════════════════════════════════════
#  DB HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_conn():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn


def get_latest_feature_date() -> str:
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM features_fused;")
            row = cur.fetchone()
        conn.close()
        return str(row[0]) if row and row[0] else str(date.today() - timedelta(days=1))
    except Exception:
        return str(date.today() - timedelta(days=1))


def get_latest_ohlcv_date() -> date:
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM daily_ohlcv;")
            row = cur.fetchone()
        conn.close()
        return row[0] if row and row[0] else date(2019, 1, 1)
    except Exception:
        return date(2019, 1, 1)


def get_latest_embedding_date() -> date:
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM backbone_embeddings;")
            row = cur.fetchone()
        conn.close()
        return row[0] if row and row[0] else date(2019, 1, 1)
    except Exception:
        return date(2019, 1, 1)


def db_status() -> dict:
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM daily_ohlcv;")
            ohlcv = cur.fetchone()[0]
            cur.execute("SELECT MAX(date) FROM features_fused;")
            feat  = cur.fetchone()[0]
            cur.execute("SELECT MAX(date) FROM backbone_embeddings;")
            emb   = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM trade_log "
                "WHERE paper_mode=TRUE AND exit_time IS NULL;"
            )
            open_pos = cur.fetchone()[0]
        conn.close()
        return {
            "ohlcv"       : str(ohlcv),
            "features"    : str(feat),
            "embeddings"  : str(emb),
            "open_positions": int(open_pos or 0),
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
#  SUBPROCESS RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run(module: str, *args, label: str = "", timeout: int = 600) -> bool:
    label = label or module
    cmd   = [str(VENV_PY), "-m", module] + list(args)
    logger.info(f"  → {label}")
    try:
        result = subprocess.run(
            cmd,
            cwd     = str(ROOT),
            timeout = timeout,
            env     = {**os.environ, "PYTHONPATH": str(ROOT)},
        )
        if result.returncode == 0:
            logger.success(f"  ✓ {label}")
            return True
        else:
            logger.error(f"  ✗ {label} (code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ {label} timed out")
        return False
    except Exception as e:
        logger.error(f"  ✗ {label}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════
#  PAPER TRADING LOGIC (backtest-faithful, embedded directly)
# ══════════════════════════════════════════════════════════════════════════

def _load_ppo_model():
    """Loads PPO model from checkpoint."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Complete Phase 3 training first."
        )

    ckpt  = torch.load(str(CHECKPOINT_PATH), map_location="cpu",
                       weights_only=False)
    state = ckpt.get("ppo_policy_state", {})
    w0    = state.get("mlp_extractor.policy_net.0.weight")
    w2    = state.get("mlp_extractor.policy_net.2.weight")
    od    = int(w0.shape[1]) if w0 is not None else OBS_DIM
    h1    = int(w0.shape[0]) if w0 is not None else 512
    h2    = int(w2.shape[0]) if w2 is not None else 256

    def _fake():
        e = gym.Env()
        e.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(od,), dtype=np.float32
        )
        e.action_space = gym.spaces.Discrete(5)
        e.reset = lambda **kw: (np.zeros(od, np.float32), {})
        e.step  = lambda a: (np.zeros(od, np.float32), 0.0, True, False, {})
        return e

    vec = DummyVecEnv([_fake])
    m   = PPO("MlpPolicy", vec, device="cpu", verbose=0,
               policy_kwargs=dict(net_arch=[h1, h2]))
    vec.close()
    m.policy.load_state_dict(state, strict=True)
    m.policy = m.policy.to("cpu")
    m.policy.eval()

    meta = ckpt.get("metadata", {})
    logger.success(
        f"PPO loaded | obs={od} | arch=[{h1},{h2}] | "
        f"val_sharpe={meta.get('val_sharpe',0):.3f} | "
        f"val_cagr={meta.get('val_cagr',0):.1%}"
    )
    return m


def _get_embeddings(target_date: str) -> dict:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT symbol, embedding FROM backbone_embeddings
            WHERE date = %s;
        """, (target_date,))
        rows = cur.fetchall()
        if not rows:
            cur.execute("""
                SELECT symbol, embedding FROM backbone_embeddings
                WHERE date = (
                    SELECT MAX(date) FROM backbone_embeddings
                    WHERE date <= %s
                );
            """, (target_date,))
            rows = cur.fetchall()
    conn.close()
    return {sym: np.array(emb, dtype=np.float32)
            for sym, emb in rows if emb}


def _get_open_positions(conn) -> list:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, signal_id, symbol, entry_price, quantity,
                   tp_price, sl_price, hold_days, confidence_score
            FROM trade_log
            WHERE paper_mode=TRUE AND exit_time IS NULL
            ORDER BY entry_time;
        """)
        return cur.fetchall()


def _get_close_price(conn, symbol: str, target_date: str) -> float:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT close FROM daily_ohlcv
            WHERE symbol=%s AND date<=%s
            ORDER BY date DESC LIMIT 1;
        """, (symbol, target_date))
        row = cur.fetchone()
    return float(row[0]) if row else 0.0


def _get_portfolio_value(conn, positions: list, target_date: str) -> float:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COALESCE(SUM(realised_pnl),0)
            FROM trade_log
            WHERE paper_mode=TRUE AND exit_time IS NOT NULL;
        """)
        realised = float(cur.fetchone()[0] or 0)
    invested = sum(float(p[3]) * int(p[4]) for p in positions)
    cash     = INITIAL_CAPITAL + realised - invested
    mtm      = sum(
        max(_get_close_price(conn, p[2], target_date), float(p[3])) * int(p[4])
        for p in positions
    )
    return cash + mtm


def _get_mds(conn, target_date: str) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ROUND(AVG(f10_mds_continuous))::int
            FROM features_fused WHERE date=%s;
        """, (target_date,))
        row = cur.fetchone()
    return int(row[0] or 0) if row and row[0] is not None else 0


def _build_obs(embeddings, positions, portfolio_value,
               cash, n_pos, heat, drawdown, mds, trades_month) -> np.ndarray:
    """Builds 5896-dim observation — exactly as GodsEyeEnv did in training."""
    emb_matrix = np.zeros((N_STOCKS, EMBEDDING_DIM), dtype=np.float32)
    for i, sym in enumerate(sorted(embeddings.keys())[:N_STOCKS]):
        emb_matrix[i] = embeddings[sym]

    max_val    = max(INITIAL_CAPITAL, portfolio_value, 1.0)
    port_state = np.array([
        portfolio_value / max_val,
        cash / max_val,
        n_pos / MAX_POSITIONS,
        heat,
        drawdown,
        (mds + 3) / 6,
        trades_month / MAX_TRADES_MONTH,
        (portfolio_value - INITIAL_CAPITAL) / max(INITIAL_CAPITAL, 1.0),
    ], dtype=np.float32)

    return np.concatenate([emb_matrix.flatten(), port_state])


def _compute_kelly_qty(entry, tp, sl, conf, pv, cash, mds, heat, dd) -> int:
    """Fractional Kelly sizing — mirrors PositionSizer exactly."""
    tp_pct = max((tp - entry) / entry, 0.01)
    sl_pct = max((entry - sl) / entry, 0.005)
    b      = tp_pct / sl_pct
    p      = np.clip(0.55 + (conf - 0.55) * (0.72 - 0.55) / 0.45, 0.50, 0.75)
    kelly  = max(0.0, (b * p - (1 - p)) / b)
    qk     = kelly * 0.25

    conf_s = (1.0 if conf >= 0.85
              else 0.75 + (conf - 0.70) * 1.667 if conf >= 0.70
              else 0.50 + (conf - 0.55) * 1.667)
    mds_m  = {3:1.5,2:1.2,1:1.0,0:1.0,-1:0.8,-2:0.5,-3:0.0}.get(mds, 1.0)
    dd_m   = (1.0 if dd < 0.05 else
              0.75 if dd < 0.08 else
              0.50 if dd < 0.10 else 0.25)

    pct    = np.clip(qk * conf_s * mds_m * dd_m, 0.05, 0.25)
    remain = max(0.0, 0.06 - heat)
    if sl_pct > 0:
        pct = min(pct, remain / sl_pct)
    pct    = np.clip(pct, 0.05, 0.25)
    invest = min(pct * pv, cash * 0.98)
    return int(invest // entry) if invest >= 5000 and entry > 0 else 0


def _check_exits(conn, positions, target_date):
    """Checks TP/SL/max-hold exactly as GodsEyeEnv did."""
    to_close = []
    for pos in positions:
        pid, sid, sym, entry, qty, tp, sl, days, conf = pos
        price = _get_close_price(conn, sym, target_date)
        if price <= 0:
            continue
        entry = float(entry); tp = float(tp); sl = float(sl)
        days  = int(days or 0)

        if   price >= tp:           to_close.append((pid, sym, tp,    "CLOSED_TP",       entry, qty))
        elif price <= sl:           to_close.append((pid, sym, sl,    "CLOSED_SL",       entry, qty))
        elif days  >= 15:           to_close.append((pid, sym, price, "CLOSED_MAX_HOLD", entry, qty))

    return to_close


def paper_trading_step(target_date: str = "", model=None):
    """
    ONE paper trading step — exactly replicates GodsEyeEnv.step().

    Entry price = official day close (same as backtest).
    Called ONCE per trading day after evening pipeline completes.
    """
    if model is None:
        model = _load_ppo_model()

    conn = get_conn()

    if not target_date:
        target_date = get_latest_feature_date()

    logger.info("═" * 60)
    logger.info(f"PAPER TRADING STEP | {target_date}")
    logger.info("═" * 60)

    # Load embeddings
    embeddings = _get_embeddings(target_date)
    if not embeddings:
        logger.error(
            f"No embeddings for {target_date}. "
            "Run precompute_embeddings first."
        )
        conn.close()
        return

    # Portfolio state
    positions = _get_open_positions(conn)
    pv        = _get_portfolio_value(conn, positions, target_date)
    invested  = sum(float(p[3]) * int(p[4]) for p in positions)
    cash      = max(0.0, pv - invested)
    n_pos     = len(positions)
    mds       = _get_mds(conn, target_date)

    # Portfolio heat
    heat = 0.0
    for p in positions:
        e = float(p[3]); s = float(p[6]); q = int(p[4])
        sl_pct = (e - s) / e if e > 0 else 0
        heat  += sl_pct * (e * q / max(pv, 1))
    heat = min(heat, 1.0)

    # Drawdown
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COALESCE(
                (%s - MAX(running_total)) / NULLIF(MAX(running_total), 0),
                0
            )
            FROM (
                SELECT SUM(realised_pnl) OVER (ORDER BY exit_time)
                       + %s AS running_total
                FROM trade_log
                WHERE paper_mode=TRUE AND exit_time IS NOT NULL
            ) t;
        """, (pv, INITIAL_CAPITAL))
        dd = abs(float(cur.fetchone()[0] or 0))

    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM trade_log
            WHERE paper_mode=TRUE
              AND entry_time >= date_trunc('month', CURRENT_DATE);
        """)
        trades_month = int(cur.fetchone()[0] or 0)

    logger.info(
        f"PV=₹{pv:,.0f} | Cash=₹{cash:,.0f} | "
        f"Pos={n_pos}/{MAX_POSITIONS} | Heat={heat:.2%} | "
        f"DD={dd:.2%} | MDS={mds} | Trades/mo={trades_month}"
    )

    # Build observation and get PPO decision
    obs     = _build_obs(embeddings, positions, pv, cash,
                         n_pos, heat, dd, mds, trades_month)
    obs_t   = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        action_arr, _ = model.predict(obs, deterministic=False)
        action_idx    = int(action_arr)
        features      = model.policy.extract_features(
            obs_t, model.policy.features_extractor
        )
        latent_pi, _  = model.policy.mlp_extractor(features)
        logits         = model.policy.action_net(latent_pi)
        probs          = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
        confidence     = float(probs[action_idx])

    action_name = ACTIONS.get(action_idx, "HOLD")
    logger.info(
        f"PPO → {action_name} | conf={confidence:.3f} | "
        f"probs={[f'{p:.3f}' for p in probs]}"
    )

    # ── Check exits FIRST (same order as GodsEyeEnv) ──────────────────────
    exits = _check_exits(conn, positions, target_date)
    for pid, sym, exit_px, reason, entry_px, qty in exits:
        pnl = (exit_px - entry_px) * int(qty)
        conn.execute("""
            UPDATE trade_log SET
                exit_price=%(ep)s, exit_time=NOW(),
                exit_reason=%(r)s, realised_pnl=%(pnl)s
            WHERE id=%(id)s;
        """) if False else None

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trade_log SET
                    exit_price=%s, exit_time=NOW(),
                    exit_reason=%s, realised_pnl=%s
                WHERE id=%s;
            """, (exit_px, reason, pnl, pid))

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} EXIT {sym:12s} | {reason:16s} | "
            f"₹{exit_px:.2f} | pnl=₹{pnl:+,.0f} "
            f"({(exit_px-entry_px)/max(entry_px,1):+.2%})"
        )

    # Refresh after exits
    positions = _get_open_positions(conn)
    n_pos     = len(positions)

    # ── Entry decision ─────────────────────────────────────────────────────
    if action_name in ("BUY", "STRONG_BUY"):
        blocked = ""
        if n_pos >= MAX_POSITIONS:          blocked = f"RC-08: {MAX_POSITIONS} positions"
        elif trades_month >= MAX_TRADES_MONTH: blocked = "RC-07: monthly limit"
        elif mds == -3:                     blocked = "RC-06: MDS=-3"
        elif confidence < MIN_CONFIDENCE:   blocked = f"confidence {confidence:.3f} < {MIN_CONFIDENCE}"
        elif heat >= 0.06:                  blocked = f"heat {heat:.2%} at max"

        if blocked:
            logger.info(f"Entry blocked: {blocked}")
        else:
            held = {p[2] for p in positions}
            with conn.cursor() as cur:
                cur.execute("""
                                SELECT ff.symbol,
                                       ff.f00_trend_score,
                                       fv.swing_tp_pct,
                                       fv.swing_sl_pct
                                FROM features_fused ff
                                JOIN features_volatility fv
                                  ON fv.symbol = ff.symbol AND fv.date = ff.date
                                WHERE ff.date = %s
                                  AND ff.symbol NOT IN %s
                                ORDER BY ff.f00_trend_score DESC
                                LIMIT 5;
                            """, (target_date,
                                  tuple(held) if held else ("__NONE__",)))
                candidates = cur.fetchall()

            if not candidates:
                logger.info("No candidate stocks available.")
            else:
                sym, trend, swing_tp_raw, swing_sl_raw = candidates[0]
                entry_price = _get_close_price(conn, sym, target_date)

                if entry_price <= 0:
                    logger.warning(f"No price for {sym}")
                else:
                    # swing_tp_pct / swing_sl_pct are stored as percentage
                    # NUMBERS (e.g. 7.30 means 7.30%), not fractions.
                    # Convert to fraction by dividing by 100.
                    tp_pct = float(swing_tp_raw or 4.0) / 100.0
                    sl_pct = float(swing_sl_raw or 1.5) / 100.0

                    # Sanity clamp — these columns are purpose-built and
                    # should already be sane, but guard against any future
                    # pipeline regression producing an unusable trade.
                    if tp_pct <= 0 or tp_pct > 0.15:
                        logger.warning(
                            f"{sym}: swing_tp_pct={swing_tp_raw} out of "
                            f"realistic range — using 4% default."
                        )
                        tp_pct = 0.04
                    if sl_pct <= 0 or sl_pct > 0.08:
                        logger.warning(
                            f"{sym}: swing_sl_pct={swing_sl_raw} out of "
                            f"realistic range — using 1.5% default."
                        )
                        sl_pct = 0.015

                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)

                    qty = _compute_kelly_qty(
                        entry_price, tp_price, sl_price,
                        confidence, pv, cash, mds, heat, dd
                    )

                    if qty > 0:
                        sig_id = f"PT-{uuid.uuid4().hex[:8].upper()}"
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO trade_log (
                                    signal_id, order_id, symbol,
                                    mode, side,
                                    entry_price, quantity, position_value,
                                    tp_price, sl_price, entry_time,
                                    confidence_score, paper_mode, hold_days
                                ) VALUES (
                                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),%s,TRUE,0
                                );
                            """, (
                                sig_id, f"PAPER-{sig_id}", sym,
                                "swing", "BUY",
                                entry_price, qty, entry_price * qty,
                                tp_price, sl_price, confidence,
                            ))
                        logger.success(
                            f"📈 ENTRY {sym:12s} | {action_name:12s} | "
                            f"qty={qty} @ ₹{entry_price:.2f} | "
                            f"TP=₹{tp_price:.2f} (+{tp_pct:.1%}) | "
                            f"SL=₹{sl_price:.2f} (-{sl_pct:.1%}) | "
                            f"conf={confidence:.3f}"
                        )
                    else:
                        logger.info(f"Kelly sizing = 0 for {sym}")
    else:
        logger.info(f"{action_name} — no new entry today.")

    # Increment hold days
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE trade_log SET hold_days = COALESCE(hold_days,0) + 1
            WHERE paper_mode=TRUE AND exit_time IS NULL;
        """)

    conn.close()


# ══════════════════════════════════════════════════════════════════════════
#  SCHEDULED TASKS
# ══════════════════════════════════════════════════════════════════════════

def task_auth(state: dict):
    if is_done(state, "auth"):
        return
    logger.info("═" * 50)
    logger.info("AUTH CHECK")
    run("data.ingestion.kite_feed", "--mode", "auth",
        label="Kite auth", timeout=120)
    mark_done(state, "auth")


def task_morning_check(state: dict):
    """
    Checks open positions against overnight data.
    Uses latest available close — NOT a new entry signal.
    """
    if is_done(state, "morning_check"):
        return
    logger.info("═" * 50)
    logger.info("MORNING CHECK — overnight black swan monitor")

    # Auto-detect and fill any data gaps first
    last_ohlcv = get_latest_ohlcv_date()
    today      = date.today()
    gap        = sum(
        1 for i in range(1, (today - last_ohlcv).days)
        if (last_ohlcv + timedelta(days=i)).weekday() < 5
    )
    if gap > 1:
        logger.warning(f"Gap of {gap} trading days detected — backfilling...")
        run("data.ingestion.nse_bhavcopy",
            "--mode", "backfill",
            "--start", str(last_ohlcv + timedelta(days=1)),
            label="OHLCV backfill", timeout=1800)

    # Check if any open positions hit TP/SL overnight
    conn    = get_conn()
    target  = get_latest_feature_date()
    positions = _get_open_positions(conn)
    if positions:
        exits = _check_exits(conn, positions, target)
        if exits:
            logger.warning(
                f"Overnight exits detected: "
                f"{[e[1] for e in exits]}"
            )
            for pid, sym, exit_px, reason, entry_px, qty in exits:
                pnl = (exit_px - entry_px) * int(qty)
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE trade_log SET
                            exit_price=%s, exit_time=NOW(),
                            exit_reason=%s, realised_pnl=%s
                        WHERE id=%s;
                    """, (exit_px, reason, pnl, pid))
                logger.info(f"Morning exit: {sym} {reason} pnl=₹{pnl:+,.0f}")
        else:
            logger.info("No overnight exits. All positions intact.")
    else:
        logger.info("No open positions to check.")
    conn.close()
    mark_done(state, "morning_check")


def task_monitor_once():
    """
    Lightweight position monitor — checks TP/SL without reloading model.
    Called every MONITOR_INTERVAL seconds during market hours.
    """
    import time as _time
    t0 = _time.time()

    conn      = get_conn()
    positions = _get_open_positions(conn)
    if not positions:
        conn.close()
        elapsed = _time.time() - t0
        if elapsed > 5:
            logger.warning(f"Monitor cycle took {elapsed:.1f}s with NO positions — DB may be slow")
        return

    target = get_latest_feature_date()
    exits  = _check_exits(conn, positions, target)
    for pid, sym, exit_px, reason, entry_px, qty in exits:
        pnl = (exit_px - entry_px) * int(qty)
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trade_log SET
                    exit_price=%s, exit_time=NOW(),
                    exit_reason=%s, realised_pnl=%s
                WHERE id=%s;
            """, (exit_px, reason, pnl, pid))
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} INTRADAY EXIT {sym} | {reason} | pnl=₹{pnl:+,.0f}"
        )
    conn.close()

    elapsed = _time.time() - t0
    if elapsed > 5:
        logger.warning(
            f"Monitor cycle took {elapsed:.1f}s "
            f"({len(positions)} positions) — investigate DB/network latency"
        )
    for pid, sym, exit_px, reason, entry_px, qty in exits:
        pnl = (exit_px - entry_px) * int(qty)
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trade_log SET
                    exit_price=%s, exit_time=NOW(),
                    exit_reason=%s, realised_pnl=%s
                WHERE id=%s;
            """, (exit_px, reason, pnl, pid))
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} INTRADAY EXIT {sym} | {reason} | pnl=₹{pnl:+,.0f}"
        )
    conn.close()


def task_evening_pipeline(state: dict):
    if is_done(state, "evening"):
        return
    logger.info("═" * 50)
    logger.info("EVENING PIPELINE")

    # 1. Download official close
    run("data.ingestion.nse_bhavcopy", "--mode", "daily",
        label="OHLCV daily", timeout=120)

    # 2. FII/DII final
    run("data.ingestion.fii_dii_scraper", "--type", "final",
        label="FII/DII final", timeout=120)

    # 3. Auto-detect feature recompute range
    last_ohlcv = get_latest_ohlcv_date()
    # Use whichever is earlier: last known feature date minus buffer,
    # or last OHLCV minus buffer. This ensures indicators have warmup.
    last_feat = get_latest_feature_date()
    try:
        last_feat_d = date.fromisoformat(last_feat)
    except Exception:
        last_feat_d = last_ohlcv - timedelta(days=FEATURE_LOOKBACK)

    feat_start = str(min(
        last_feat_d - timedelta(days=FEATURE_LOOKBACK),
        last_ohlcv - timedelta(days=FEATURE_LOOKBACK),
    ))
    logger.info(
        f"Recomputing features from {feat_start} "
        f"(last_feat={last_feat} last_ohlcv={last_ohlcv})"
    )

    # Run in dependency order — fusion must run last
    # Each gets the same wide start date for indicator warmup
    feature_pipeline = [
        ("features.trend", 300),
        ("features.msi", 300),
        ("features.volatility", 300),
        ("features.correlation", 300),
        ("features.fusion", 600),  # longer timeout — reads all pillars
    ]
    for module, timeout in feature_pipeline:
        success = run(module, "--mode", "all", "--start", feat_start,
                      label=module, timeout=timeout)
        if not success:
            logger.warning(
                f"{module} failed — continuing with other pillars. "
                f"Trade step will validate data before executing."
            )

    # 4. Recompute embeddings from features_fused (full 28 features)
    run("training.precompute_embeddings", "--device", "cuda",
        label="Embeddings (from features_fused)", timeout=600)

    # 5. Verify
    s = db_status()
    logger.info(
        f"Data state after pipeline: "
        f"OHLCV={s.get('ohlcv')} | "
        f"Features={s.get('features')} | "
        f"Embeddings={s.get('embeddings')}"
    )
    mark_done(state, "evening")


def task_trade_step(state: dict):
    """
    ONE paper trading step with official close data.
    This exactly replicates backtest behavior.
    Entry price = today's official close.
    """
    if is_done(state, "trade"):
        return
    logger.info("═" * 50)
    logger.info("PAPER TRADING STEP (backtest-faithful)")

    # Validate data freshness before trading
    s           = db_status()
    ohlcv_date  = s.get("ohlcv",      "")
    feat_date   = s.get("features",   "")
    emb_date    = s.get("embeddings", "")
    today_str   = str(date.today())
    yesterday   = str(date.today() - timedelta(days=1))

    # Features must be within last 2 trading days
    if feat_date < yesterday:
        logger.error(
            f"Features are stale ({feat_date}) — "
            f"evening pipeline may have failed. "
            f"Run manually: python orchestrator.py --run-now evening"
        )
        logger.error("Skipping trade step to avoid stale signal.")
        return

    # Embeddings must be within last 2 trading days
    if emb_date < yesterday:
        logger.error(
            f"Embeddings are stale ({emb_date}) — "
            f"run: python -m training.precompute_embeddings --device cuda"
        )
        logger.error("Skipping trade step to avoid wrong embeddings.")
        return

    logger.info(
        f"Data validation passed | "
        f"OHLCV={ohlcv_date} | Features={feat_date} | Embeddings={emb_date}"
    )

    model = _load_ppo_model()
    paper_trading_step(model=model)
    mark_done(state, "trade")


def task_daily_report(state: dict):
    if is_done(state, "report"):
        return
    logger.info("═" * 50)
    logger.info(f"DAILY REPORT — {date.today()}")
    logger.info("═" * 50)

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE exit_time IS NULL)          open_pos,
                COUNT(*) FILTER (WHERE exit_time::date=CURRENT_DATE) closed_today,
                COUNT(*) FILTER (WHERE exit_time::date=CURRENT_DATE
                                  AND realised_pnl>0)              wins_today,
                COALESCE(SUM(realised_pnl)
                         FILTER (WHERE exit_time::date=CURRENT_DATE), 0) pnl_today,
                COUNT(*) FILTER (WHERE exit_time IS NOT NULL)      total_closed,
                COALESCE(SUM(realised_pnl)
                         FILTER (WHERE exit_time IS NOT NULL), 0)  total_pnl,
                COUNT(*) FILTER (WHERE exit_time IS NOT NULL
                                  AND realised_pnl>0)              total_wins
            FROM trade_log WHERE paper_mode=TRUE;
        """)
        r = cur.fetchone()
        cur.execute("""
            SELECT symbol, entry_price, quantity, tp_price, sl_price,
                   hold_days, confidence_score
            FROM trade_log
            WHERE paper_mode=TRUE AND exit_time IS NULL;
        """)
        positions = cur.fetchall()
    conn.close()

    closed = int(r[1] or 0); wins = int(r[2] or 0)
    total  = int(r[4] or 0); tw   = int(r[6] or 0)

    logger.info(f"  Today P&L        : ₹{float(r[3] or 0):+,.2f}")
    logger.info(f"  Today trades     : {closed} ({wins} wins)")
    logger.info(f"  Total P&L        : ₹{float(r[5] or 0):+,.2f}")
    logger.info(f"  Total trades     : {total} | "
                f"WinRate: {tw/total*100:.1f}%" if total > 0 else
                f"  Total trades     : 0")
    logger.info(f"  Open positions   : {int(r[0] or 0)}")

    if positions:
        logger.info("  Open positions detail:")
        for p in positions:
            sym, entry, qty, tp, sl, days, conf = p
            logger.info(
                f"    {sym:12s} entry=₹{float(entry):.2f} qty={qty} "
                f"TP=₹{float(tp):.2f} SL=₹{float(sl):.2f} "
                f"day={int(days or 0)} conf={float(conf or 0):.2f}"
            )
    logger.info("═" * 50)
    mark_done(state, "report")


# ══════════════════════════════════════════════════════════════════════════
#  TIME HELPERS
# ══════════════════════════════════════════════════════════════════════════

def now_hm():
    n = datetime.now()
    return (n.hour, n.minute)


def after(h, m):  return now_hm() >= (h, m)
def before(h, m): return now_hm() <  (h, m)
def between(h1, m1, h2, m2): return (h1,m1) <= now_hm() < (h2,m2)
def is_weekday():  return date.today().weekday() < 5


def sleep_until(h, m, label=""):
    now    = datetime.now()
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target < now:
        target += timedelta(days=1)
    secs = (target - now).total_seconds()
    if secs > 0:
        logger.info(
            f"Sleeping until {h:02d}:{m:02d}"
            + (f" ({label})" if label else "")
            + f" — {secs/60:.0f} min"
        )
        time.sleep(secs)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   G.O.D.S E.Y.E — Daily Orchestrator         ║")
    logger.info("╚═══════════════════════════════════════════════╝")
    logger.info(f"Today : {date.today().strftime('%A %d %B %Y')}")

    s   = db_status()
    logger.info(
        f"DB state | OHLCV={s.get('ohlcv')} | "
        f"Features={s.get('features')} | "
        f"Embeddings={s.get('embeddings')} | "
        f"OpenPos={s.get('open_positions',0)}"
    )

    state = load_state()
    done  = [k for k, v in state.items()
             if isinstance(v, dict) and v.get("done")]
    if done:
        logger.info(f"Already done today: {done}")

    while True:
        if not is_weekday():
            logger.info("Weekend — sleeping 1 hour")
            time.sleep(3600)
            continue

        S = SCHEDULE

        # 08:00 — Auth
        if after(*S["auth"]) and before(*S["morning_check"]):
            task_auth(state)

        # 08:05 — Morning check
        elif after(*S["morning_check"]) and before(*S["market_open"]):
            task_morning_check(state)
            sleep_until(*S["market_open"], "market open")

        # 09:15–15:30 — Position monitoring (self-correcting timing)
        elif between(*S["market_open"], *S["market_close"]):
            cycle_start = datetime.now()
            logger.info(
                f"[{cycle_start.strftime('%H:%M:%S')}] "
                f"Market hours — monitoring positions..."
            )

            task_monitor_once()

            cycle_elapsed = (datetime.now() - cycle_start).total_seconds()

            if cycle_elapsed >= MONITOR_INTERVAL:
                # Cycle itself took longer than the interval —
                # skip sleeping entirely and immediately check if
                # we're still in market hours before looping again
                missed_cycles = int(cycle_elapsed // MONITOR_INTERVAL)
                logger.warning(
                    f"Monitor cycle took {cycle_elapsed:.0f}s "
                    f"(missed ~{missed_cycles} cycle(s)). "
                    f"Re-checking immediately — no sleep."
                )
                # Re-verify we're still within market hours before
                # looping back; if market has closed meanwhile,
                # the outer while loop will naturally fall through
                # to the next branch (market_close handling)
                continue
            else:
                # Normal case — sleep only the REMAINING time
                # so checks stay aligned to 60-second boundaries
                remaining = MONITOR_INTERVAL - cycle_elapsed
                time.sleep(remaining)

        # 15:31–17:59 — Post-market wait
        elif after(*S["market_close"]) and before(*S["evening_start"]):
            logger.info("Market closed. Waiting for evening pipeline...")
            sleep_until(*S["evening_start"], "evening pipeline")

        # 18:00 — Evening pipeline
        elif after(*S["evening_start"]) and before(*S["trade_step"]):
            task_evening_pipeline(state)

        # 18:45 — Paper trading step (backtest-faithful)
        elif after(*S["trade_step"]) and before(*S["daily_report"]):
            task_trade_step(state)

        # 18:50 — Daily report
        elif after(*S["daily_report"]):
            task_daily_report(state)
            logger.info("All tasks done. Sleeping until tomorrow 8 AM...")
            sleep_until(*S["sleep_until"], "tomorrow")
            state = {}
            save_state(state)

        # Before 8 AM
        else:
            logger.info(
                f"[{datetime.now().strftime('%H:%M')}] "
                f"Waiting for 8:00 AM... (5 min sleep)"
            )
            time.sleep(300)


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E Orchestrator"
    )
    parser.add_argument(
        "--run-now",
        choices=["auth","morning","evening","trade","report","status","backfill"],
        help="Force-run a specific task immediately"
    )
    parser.add_argument(
        "--days", type=int, default=5,
        help="Days to backfill (used with --run-now backfill)"
    )
    parser.add_argument(
        "--date", default="",
        help="Specific date for trade step YYYY-MM-DD"
    )
    args = parser.parse_args()

    if args.run_now:
        state = load_state()

        if args.run_now == "status":
            s = db_status()
            logger.info("─" * 50)
            logger.info(f"Date        : {date.today()}")
            logger.info(f"OHLCV       : {s.get('ohlcv')}")
            logger.info(f"Features    : {s.get('features')}")
            logger.info(f"Embeddings  : {s.get('embeddings')}")
            logger.info(f"Open pos    : {s.get('open_positions',0)}")
            logger.info(f"Done today  : {[k for k,v in state.items() if isinstance(v,dict) and v.get('done')]}")
            logger.info("─" * 50)

        elif args.run_now == "auth":
            state.pop("auth", None)
            task_auth(state)

        elif args.run_now == "morning":
            state.pop("morning_check", None)
            task_morning_check(state)

        elif args.run_now == "evening":
            state.pop("evening", None)
            task_evening_pipeline(state)

        elif args.run_now == "trade":
            state.pop("trade", None)
            model = _load_ppo_model()
            paper_trading_step(
                target_date=args.date or "",
                model=model
            )
            mark_done(state, "trade")

        elif args.run_now == "report":
            state.pop("report", None)
            task_daily_report(state)

        elif args.run_now == "backfill":
            # Run paper trading step for last N days
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT date FROM features_fused
                    ORDER BY date DESC LIMIT %s;
                """, (args.days,))
                dates = sorted([str(r[0]) for r in cur.fetchall()])
            conn.close()

            logger.info(f"Backfill: running {len(dates)} days: {dates}")
            model = _load_ppo_model()
            for d in dates:
                paper_trading_step(target_date=d, model=model)
    else:
        try:
            main()
        except KeyboardInterrupt:
            logger.info("Orchestrator stopped.")
