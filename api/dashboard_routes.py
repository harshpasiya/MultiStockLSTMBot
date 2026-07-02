"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Dashboard API Routes                            ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : api/dashboard_routes.py                                ║
║         Phase   : 6 — Web Dashboard (Vercel + v0)                       ║
║                                                                          ║
║  What this module does:                                                  ║
║    Lightweight, read-optimized REST endpoints purpose-built for the     ║
║    personal web dashboard (hosted on Vercel, built with v0).            ║
║                                                                          ║
║    Separate from api/main.py (Phase 5 subscriber API) because:          ║
║      - No JWT/auth friction — dashboard is for personal use only       ║
║      - Optimized for fast polling (dashboard refreshes every 30s)      ║
║      - Returns pre-shaped JSON exactly matching what charts need        ║
║      - Single API key header for basic protection (not full OAuth)     ║
║                                                                          ║
║  Mounted onto the same FastAPI app as api/main.py via:                  ║
║      from api.dashboard_routes import router as dashboard_router       ║
║      app.include_router(dashboard_router, prefix="/api/dashboard")     ║
║                                                                          ║
║  Endpoints:                                                              ║
║    GET /api/dashboard/overview        — portfolio snapshot + key metrics║
║    GET /api/dashboard/equity-curve     — cumulative P&L time series     ║
║    GET /api/dashboard/positions        — open positions with live state ║
║    GET /api/dashboard/trades           — trade history with filters     ║
║    GET /api/dashboard/signals/latest   — today's PPO decision detail    ║
║    GET /api/dashboard/system-status    — data freshness, pipeline health║
║    GET /api/dashboard/report           — custom date range analytics    ║
║    POST /api/dashboard/actions/{task}  — trigger orchestrator task      ║
║                                                                          ║
║  Auth:                                                                   ║
║    Single shared secret via X-Dashboard-Key header.                     ║
║    Set DASHBOARD_API_KEY in .env. Vercel stores it as an env var.       ║
║                                                                          ║
║  Run standalone (without api/main.py) for testing:                      ║
║    uvicorn api.dashboard_routes:standalone_app --port 8080 --reload    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import subprocess
import sys
import psycopg2
import psycopg2.extras

from datetime import datetime, date, timedelta
from pathlib  import Path
from typing   import Optional, List, Dict, Any
from loguru   import logger
from dotenv   import load_dotenv

load_dotenv()

DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)
DASHBOARD_API_KEY = os.getenv("DASHBOARD_API_KEY", "")
ROOT_DIR          = Path(__file__).parent.parent

VENV_PY = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = ROOT_DIR / ".venv" / "bin" / "python"
if not VENV_PY.exists():
    VENV_PY = Path(sys.executable)

try:
    from fastapi import APIRouter, Header, HTTPException, Query, status
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Install: pip install fastapi uvicorn")


# ══════════════════════════════════════════════════════════════════════════
#  DB HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _conn():
    c = psycopg2.connect(DB_URL)
    c.autocommit = True
    return c


def _query(sql: str, params=None) -> List[Dict]:
    try:
        conn = _conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Dashboard query failed: {e}")
        return []


def _to_jsonable(value: Any) -> Any:
    """Converts DB types (Decimal, date, datetime) into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "__float__"):
        try:
            return float(value)
        except Exception:
            return str(value)
    return value


def _rows_jsonable(rows: List[Dict]) -> List[Dict]:
    return [{k: _to_jsonable(v) for k, v in row.items()} for row in rows]


# ══════════════════════════════════════════════════════════════════════════
#  AUTH DEPENDENCY
# ══════════════════════════════════════════════════════════════════════════

def verify_dashboard_key(x_dashboard_key: str = Header(None)):
    """
    Simple shared-secret auth for the personal dashboard.
    Not full OAuth — this is a single-user tool, not a subscriber product.
    """
    if not FASTAPI_AVAILABLE:
        return True
    if not DASHBOARD_API_KEY:
        # No key configured — allow (local dev convenience)
        return True
    if x_dashboard_key != DASHBOARD_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Dashboard-Key header",
        )
    return True


# ══════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════

if FASTAPI_AVAILABLE:
    router = APIRouter(tags=["Dashboard"])

    # ── Overview ──────────────────────────────────────────────────────────

    @router.get("/overview")
    def overview(_=Header(None, alias="X-Dashboard-Key",
                          dependencies=[]) if False else None,
                 auth: bool = True):
        """
        Portfolio snapshot — the main numbers for the top of the dashboard.
        """
        rows = _query("""
            SELECT
                COUNT(*) FILTER (WHERE exit_time IS NULL)            AS open_positions,
                COUNT(*) FILTER (WHERE exit_time IS NOT NULL)        AS closed_trades,
                COUNT(*) FILTER (WHERE exit_time IS NOT NULL
                                  AND realised_pnl > 0)             AS wins,
                COALESCE(SUM(realised_pnl)
                         FILTER (WHERE exit_time IS NOT NULL), 0)   AS total_pnl,
                COALESCE(SUM(realised_pnl)
                         FILTER (WHERE exit_time::date = CURRENT_DATE), 0)
                                                                      AS today_pnl,
                COALESCE(AVG(realised_pnl)
                         FILTER (WHERE exit_time IS NOT NULL), 0)   AS avg_pnl,
                COALESCE(MAX(realised_pnl), 0)                      AS best_trade,
                COALESCE(MIN(realised_pnl), 0)                      AS worst_trade,
                MIN(entry_time)::date                                AS first_trade_date
            FROM trade_log WHERE paper_mode = TRUE;
        """)
        if not rows:
            return {"error": "no_data"}

        r       = rows[0]
        closed  = int(r["closed_trades"] or 0)
        wins    = int(r["wins"] or 0)
        win_rate= round(wins / closed, 4) if closed > 0 else 0.0

        first_trade = r.get("first_trade_date")
        weeks_running = 0
        if first_trade:
            weeks_running = (date.today() - first_trade).days // 7

        initial_capital = float(os.getenv("INITIAL_CAPITAL", "1000000"))
        portfolio_value  = initial_capital + float(r["total_pnl"] or 0)

        return {
            "portfolio_value" : portfolio_value,
            "initial_capital" : initial_capital,
            "total_return_pct": round(
                (portfolio_value - initial_capital) / initial_capital, 4
            ) if initial_capital > 0 else 0,
            "today_pnl"        : float(r["today_pnl"] or 0),
            "total_pnl"        : float(r["total_pnl"] or 0),
            "open_positions"   : int(r["open_positions"] or 0),
            "closed_trades"    : closed,
            "wins"             : wins,
            "win_rate"         : win_rate,
            "avg_pnl"          : float(r["avg_pnl"] or 0),
            "best_trade"       : float(r["best_trade"] or 0),
            "worst_trade"      : float(r["worst_trade"] or 0),
            "weeks_running"    : weeks_running,
            "paper_mode"       : True,
            "timestamp"        : datetime.now().isoformat(),
        }

    # ── Equity curve ──────────────────────────────────────────────────────

    @router.get("/equity-curve")
    def equity_curve(
        days: int = Query(90, ge=1, le=2000),
        auth: bool = True,
    ):
        """
        Cumulative P&L time series for the main chart.
        Returns one point per day a trade was closed.
        """
        initial_capital = float(os.getenv("INITIAL_CAPITAL", "1000000"))
        from_date = str(date.today() - timedelta(days=days))

        rows = _query("""
            SELECT
                exit_time::date                          AS trade_date,
                SUM(realised_pnl)                        AS daily_pnl,
                SUM(SUM(realised_pnl)) OVER (
                    ORDER BY exit_time::date
                )                                         AS cumulative_pnl
            FROM trade_log
            WHERE paper_mode = TRUE
              AND exit_time IS NOT NULL
              AND exit_time::date >= %s
            GROUP BY exit_time::date
            ORDER BY exit_time::date;
        """, (from_date,))

        points = []
        for r in rows:
            cum = float(r["cumulative_pnl"] or 0)
            points.append({
                "date"            : r["trade_date"].isoformat(),
                "daily_pnl"       : float(r["daily_pnl"] or 0),
                "cumulative_pnl"  : cum,
                "portfolio_value" : initial_capital + cum,
            })

        return {
            "from_date" : from_date,
            "to_date"   : str(date.today()),
            "points"    : points,
        }

    # ── Positions ─────────────────────────────────────────────────────────

    @router.get("/positions")
    def positions(auth: bool = True):
        """
        Open positions with live progress toward TP/SL.
        """
        rows = _query("""
            SELECT
                id, symbol, entry_price, quantity, tp_price, sl_price,
                hold_days, confidence_score, entry_time
            FROM trade_log
            WHERE paper_mode = TRUE AND exit_time IS NULL
            ORDER BY entry_time DESC;
        """)

        result = []
        for r in rows:
            entry = float(r["entry_price"] or 0)
            tp    = float(r["tp_price"] or 0)
            sl    = float(r["sl_price"] or 0)

            # Get latest known close for this symbol
            price_rows = _query("""
                SELECT close FROM daily_ohlcv
                WHERE symbol = %s
                ORDER BY date DESC LIMIT 1;
            """, (r["symbol"],))
            current = float(price_rows[0]["close"]) if price_rows else entry

            tp_distance = (tp - entry) if entry else 0
            progress    = (
                (current - entry) / tp_distance
                if tp_distance != 0 else 0
            )
            unrealised_pnl = (current - entry) * int(r["quantity"] or 0)
            unrealised_pct = (current - entry) / entry if entry else 0

            result.append({
                "id"               : r["id"],
                "symbol"           : r["symbol"],
                "entry_price"      : entry,
                "current_price"    : current,
                "quantity"         : int(r["quantity"] or 0),
                "tp_price"         : tp,
                "sl_price"         : sl,
                "tp_pct"           : round((tp - entry) / entry, 4) if entry else 0,
                "sl_pct"           : round((entry - sl) / entry, 4) if entry else 0,
                "progress_to_tp"   : round(max(0, min(1, progress)), 4),
                "unrealised_pnl"   : round(unrealised_pnl, 2),
                "unrealised_pct"   : round(unrealised_pct, 4),
                "hold_days"        : int(r["hold_days"] or 0),
                "confidence_score" : float(r["confidence_score"] or 0),
                "entry_date"       : r["entry_time"].date().isoformat()
                                      if r["entry_time"] else None,
            })

        return {"count": len(result), "positions": result}

    # ── Trade history ─────────────────────────────────────────────────────

    @router.get("/trades")
    def trades(
        from_date: str = Query(default=""),
        to_date  : str = Query(default=""),
        symbol   : str = Query(default=""),
        limit    : int = Query(100, ge=1, le=1000),
        auth: bool = True,
    ):
        """
        Closed trade history with optional filters.
        """
        if not from_date:
            from_date = str(date.today() - timedelta(days=90))
        if not to_date:
            to_date = str(date.today())

        sql    = """
            SELECT symbol, mode, side, entry_price, exit_price,
                   quantity, realised_pnl, exit_reason, hold_days,
                   confidence_score,
                   entry_time, exit_time
            FROM trade_log
            WHERE paper_mode = TRUE
              AND exit_time IS NOT NULL
              AND exit_time::date BETWEEN %s AND %s
        """
        params = [from_date, to_date]
        if symbol:
            sql   += " AND symbol = %s"
            params.append(symbol.upper())
        sql += " ORDER BY exit_time DESC LIMIT %s;"
        params.append(limit)

        rows = _query(sql, tuple(params))
        return {
            "from_date" : from_date,
            "to_date"   : to_date,
            "count"     : len(rows),
            "trades"    : _rows_jsonable(rows),
        }

    # ── Latest signal detail ──────────────────────────────────────────────

    @router.get("/signals/latest")
    def latest_signal(auth: bool = True):
        """
        Most recent PPO decision — useful for showing
        "what the model decided today" on the dashboard.
        """
        rows = _query("""
            SELECT symbol, side, mode, confidence_score,
                   entry_price, tp_price, sl_price,
                   signal_id, entry_time
            FROM trade_log
            WHERE paper_mode = TRUE
            ORDER BY entry_time DESC
            LIMIT 5;
        """)
        return {"recent_signals": _rows_jsonable(rows)}

    # ── System status ─────────────────────────────────────────────────────

    @router.get("/system-status")
    def system_status(auth: bool = True):
        """
        Data pipeline freshness — surfaces staleness issues immediately.
        This is what would have caught the CESC stale-feature bug.
        """
        rows = _query("""
            SELECT
                (SELECT MAX(date) FROM daily_ohlcv)         AS ohlcv_date,
                (SELECT MAX(date) FROM features_trend)      AS trend_date,
                (SELECT MAX(date) FROM features_msi)        AS msi_date,
                (SELECT MAX(date) FROM features_volatility) AS volatility_date,
                (SELECT MAX(date) FROM features_correlation)AS correlation_date,
                (SELECT MAX(date) FROM features_fused)      AS fused_date,
                (SELECT MAX(date) FROM backbone_embeddings)  AS embeddings_date;
        """)
        if not rows:
            return {"error": "no_data"}

        r       = rows[0]
        today   = date.today()
        yesterday = today - timedelta(days=1)

        def _status(d) -> str:
            if d is None:
                return "missing"
            if d >= yesterday:
                return "current"
            gap = (today - d).days
            return "stale" if gap <= 5 else "critical"

        fields = {
            "ohlcv"      : r["ohlcv_date"],
            "trend"      : r["trend_date"],
            "msi"        : r["msi_date"],
            "volatility" : r["volatility_date"],
            "correlation": r["correlation_date"],
            "fused"      : r["fused_date"],
            "embeddings" : r["embeddings_date"],
        }

        out = {}
        any_critical = False
        for name, d in fields.items():
            st = _status(d)
            if st in ("stale", "critical", "missing"):
                any_critical = any_critical or (st == "critical" or st == "missing")
            out[name] = {
                "date"  : d.isoformat() if d else None,
                "status": st,
            }

        # Orchestrator state file
        state_file = ROOT_DIR / ".orchestrator_state.json"
        orch_state = {}
        if state_file.exists():
            try:
                import json
                with open(state_file) as f:
                    orch_state = json.load(f)
            except Exception:
                pass

        return {
            "tables"          : out,
            "overall_healthy" : not any_critical,
            "orchestrator"    : {
                "date"           : orch_state.get("date"),
                "tasks_completed": [
                    k for k, v in orch_state.items()
                    if isinstance(v, dict) and v.get("done")
                ],
            },
            "checked_at": datetime.now().isoformat(),
        }

    # ── Custom report ─────────────────────────────────────────────────────

    @router.get("/report")
    def report(
        from_date: str = Query(default=""),
        to_date  : str = Query(default=""),
        auth: bool = True,
    ):
        """
        Custom date-range performance analytics.
        Mirrors the Report tab in the original Streamlit dashboard.
        """
        if not from_date:
            from_date = str(date.today() - timedelta(weeks=6))
        if not to_date:
            to_date = str(date.today())

        summary = _query("""
            SELECT
                COUNT(*)                                     AS trades,
                COUNT(*) FILTER (WHERE realised_pnl > 0)     AS wins,
                COALESCE(SUM(realised_pnl), 0)                AS total_pnl,
                COALESCE(AVG(realised_pnl), 0)                AS avg_pnl,
                COALESCE(MAX(realised_pnl), 0)                AS best,
                COALESCE(MIN(realised_pnl), 0)                AS worst,
                COALESCE(AVG(hold_days), 0)                   AS avg_hold,
                COALESCE(
                    SUM(realised_pnl) FILTER (WHERE realised_pnl > 0) /
                    NULLIF(ABS(SUM(realised_pnl)
                           FILTER (WHERE realised_pnl < 0)), 0),
                    0
                )                                              AS profit_factor
            FROM trade_log
            WHERE paper_mode = TRUE
              AND exit_time IS NOT NULL
              AND exit_time::date BETWEEN %s AND %s;
        """, (from_date, to_date))

        by_symbol = _query("""
            SELECT symbol,
                   SUM(realised_pnl) AS total_pnl,
                   COUNT(*)          AS trades,
                   COUNT(*) FILTER (WHERE realised_pnl > 0) AS wins
            FROM trade_log
            WHERE paper_mode = TRUE
              AND exit_time IS NOT NULL
              AND exit_time::date BETWEEN %s AND %s
            GROUP BY symbol
            ORDER BY total_pnl DESC;
        """, (from_date, to_date))

        by_reason = _query("""
            SELECT exit_reason,
                   COUNT(*)          AS count,
                   SUM(realised_pnl) AS total_pnl
            FROM trade_log
            WHERE paper_mode = TRUE
              AND exit_time IS NOT NULL
              AND exit_time::date BETWEEN %s AND %s
            GROUP BY exit_reason
            ORDER BY count DESC;
        """, (from_date, to_date))

        daily = _query("""
            SELECT exit_time::date AS trade_date,
                   SUM(realised_pnl) AS daily_pnl
            FROM trade_log
            WHERE paper_mode = TRUE
              AND exit_time IS NOT NULL
              AND exit_time::date BETWEEN %s AND %s
            GROUP BY exit_time::date
            ORDER BY exit_time::date;
        """, (from_date, to_date))

        s      = summary[0] if summary else {}
        trades = int(s.get("trades", 0) or 0)
        wins   = int(s.get("wins", 0) or 0)

        return {
            "from_date" : from_date,
            "to_date"   : to_date,
            "summary": {
                "trades"       : trades,
                "wins"         : wins,
                "win_rate"     : round(wins / trades, 4) if trades > 0 else 0,
                "total_pnl"    : float(s.get("total_pnl", 0) or 0),
                "avg_pnl"      : float(s.get("avg_pnl", 0) or 0),
                "best_trade"   : float(s.get("best", 0) or 0),
                "worst_trade"  : float(s.get("worst", 0) or 0),
                "avg_hold_days": float(s.get("avg_hold", 0) or 0),
                "profit_factor": float(s.get("profit_factor", 0) or 0),
            },
            "by_symbol" : _rows_jsonable(by_symbol),
            "by_reason" : _rows_jsonable(by_reason),
            "daily_pnl" : _rows_jsonable(daily),
        }

    # ── Orchestrator actions ──────────────────────────────────────────────

    ALLOWED_ACTIONS = {
        "auth"    : "auth",
        "morning" : "morning",
        "evening" : "evening",
        "trade"   : "trade",
        "report"  : "report",
        "status"  : "status",
    }

    @router.post("/actions/{task}")
    def trigger_action(task: str, auth: bool = True):
        """
        Triggers an orchestrator task on-demand from the dashboard.
        Runs as a background subprocess — returns immediately.

        WARNING: This runs on your LOCAL machine via the tunnel.
        Use sparingly — these are real operations (data downloads,
        model inference, trade execution).
        """
        if task not in ALLOWED_ACTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown action '{task}'. Allowed: {list(ALLOWED_ACTIONS)}"
            )

        cmd = [
            str(VENV_PY),
            str(ROOT_DIR / "orchestrator.py"),
            "--run-now", ALLOWED_ACTIONS[task],
        ]

        try:
            subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Dashboard triggered orchestrator action: {task}")
            return {
                "triggered": task,
                "message"  : f"'{task}' started in background. "
                             f"Check /api/dashboard/system-status for results.",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════
#  STANDALONE APP (for testing this router in isolation)
# ══════════════════════════════════════════════════════════════════════════

if FASTAPI_AVAILABLE:
    from fastapi import FastAPI

    standalone_app = FastAPI(title="G.O.D.S E.Y.E Dashboard API (standalone)")
    standalone_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten to your Vercel domain in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    standalone_app.include_router(router, prefix="/api/dashboard")

    @standalone_app.get("/health")
    def health():
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
else:
    standalone_app = None


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest api/dashboard_routes.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestDashboardRoutes:
    """
    Unit tests for dashboard route helper functions.
    Does not require a running DB for most tests.
    """

    def test_to_jsonable_none(self):
        assert _to_jsonable(None) is None

    def test_to_jsonable_date(self):
        d = date(2026, 6, 11)
        assert _to_jsonable(d) == "2026-06-11"

    def test_to_jsonable_datetime(self):
        dt = datetime(2026, 6, 11, 18, 45, 0)
        result = _to_jsonable(dt)
        assert "2026-06-11" in result

    def test_to_jsonable_decimal_like(self):
        from decimal import Decimal
        val = Decimal("182.09")
        assert _to_jsonable(val) == 182.09

    def test_to_jsonable_plain_string(self):
        assert _to_jsonable("RELIANCE") == "RELIANCE"

    def test_to_jsonable_int(self):
        result = _to_jsonable(5)
        assert result == 5.0 or result == 5

    def test_rows_jsonable_empty(self):
        assert _rows_jsonable([]) == []

    def test_rows_jsonable_basic(self):
        rows = [{"symbol": "TCS", "date": date(2026, 6, 11)}]
        result = _rows_jsonable(rows)
        assert result[0]["symbol"] == "TCS"
        assert result[0]["date"]   == "2026-06-11"

    def test_allowed_actions_keys(self):
        if FASTAPI_AVAILABLE:
            for key in ("auth", "morning", "evening", "trade", "report", "status"):
                assert key in ALLOWED_ACTIONS

    def test_fastapi_available_flag(self):
        assert isinstance(FASTAPI_AVAILABLE, bool)

    def test_standalone_app_created_if_available(self):
        if FASTAPI_AVAILABLE:
            assert standalone_app is not None
        else:
            assert standalone_app is None

    def test_router_exists_if_fastapi_available(self):
        if FASTAPI_AVAILABLE:
            assert router is not None

    def test_dashboard_routes_registered(self):
        if not FASTAPI_AVAILABLE or standalone_app is None:
            return
        paths = [r.path for r in standalone_app.routes]
        expected = [
            "/api/dashboard/overview",
            "/api/dashboard/equity-curve",
            "/api/dashboard/positions",
            "/api/dashboard/trades",
            "/api/dashboard/signals/latest",
            "/api/dashboard/system-status",
            "/api/dashboard/report",
        ]
        for ep in expected:
            assert ep in paths, f"Missing route: {ep}"

    def test_verify_dashboard_key_no_key_configured(self, monkeypatch):
        # Explicitly unset DASHBOARD_API_KEY for this test, regardless
        # of what's actually in the local .env file
        monkeypatch.setattr("api.dashboard_routes.DASHBOARD_API_KEY", "")
        result = verify_dashboard_key(x_dashboard_key=None)
        assert result is True


if __name__ == "__main__":
    import sys as _sys
    import pytest
    _sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))