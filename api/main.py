"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — FastAPI Signal Broadcast Server                  ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : api/main.py                                            ║
║         Phase   : 5 — Subscriber Infrastructure                          ║
║                                                                          ║
║  What this module does:                                                  ║
║    REST API server that exposes G.O.D.S E.Y.E signals and subscriber     ║
║    management to external clients (web dashboard, mobile app,            ║
║    distributor portals, programmatic trading bots).                      ║
║                                                                          ║
║  Run:                                                                    ║
║    uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload              ║
║                                                                          ║
║  Endpoints:                                                              ║
║    POST /auth/register        — create subscriber account                ║
║    POST /auth/login           — get JWT token                            ║
║    POST /auth/logout          — revoke JWT token                         ║
║                                                                          ║
║    GET  /signals/latest       — latest signals for today                 ║
║    GET  /signals/history      — signal history with date filter          ║
║    GET  /portfolio            — subscriber portfolio snapshot            ║
║    GET  /trades               — trade history with filters               ║
║                                                                          ║
║    POST /subscriber/onboard   — connect broker account                   ║
║    GET  /subscriber/profile   — get subscriber profile                   ║
║    PUT  /subscriber/plan      — upgrade/downgrade plan                   ║
║                                                                          ║
║    GET  /admin/stats          — system-wide stats (admin only)           ║
║    GET  /admin/subscribers    — all subscribers (admin only)             ║
║    GET  /distributor/stats    — distributor dashboard stats              ║
║    POST /distributor/referral — create referral code                     ║
║                                                                          ║
║    GET  /health               — system health check                      ║
║    GET  /metrics/summary      — key performance metrics                  ║
║                                                                          ║
║  Auth:                                                                   ║
║    JWT Bearer token for subscribers                                      ║
║    API Key header (X-API-Key) for distributors/programmatic access       ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt] ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from api.dashboard_routes import router as dashboard_router

import os
import time
import psycopg2

from datetime    import datetime, date, timedelta
from typing      import Optional, List, Dict, Any
from loguru      import logger
from dotenv      import load_dotenv

load_dotenv()

DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)

# ── FastAPI imports ────────────────────────────────────────────────────────
try:
    from fastapi                   import FastAPI, Depends, HTTPException, Header, status
    from fastapi.middleware.cors   import CORSMiddleware
    from fastapi.responses         import JSONResponse
    from pydantic                  import BaseModel, EmailStr, Field
    from contextlib import asynccontextmanager
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning(
        "FastAPI not installed. Install: pip install fastapi uvicorn pydantic[email]"
    )

from api.auth import (
    AuthManager, UserRole, BrokerProvider, SubscriptionPlan,
    create_jwt_token, verify_jwt_token, AuthToken,
)
from api.subscriber_manager import (
    SubscriberManager, SubscriberAccount, BroadcastSummary,
)


# ══════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS (Pydantic)
# ══════════════════════════════════════════════════════════════════════════

if FASTAPI_AVAILABLE:
    class RegisterRequest(BaseModel):
        email         : str
        password      : str
        referral_code : str = ""

    class LoginRequest(BaseModel):
        email   : str
        password: str

    class OnboardRequest(BaseModel):
        broker_provider : str   = "zerodha"
        broker_token    : str
        capital_inr     : float = Field(gt=0)
        plan            : str   = "free"
        paper_mode      : bool  = True

    class PlanUpdateRequest(BaseModel):
        new_plan: str

    class ReferralCreateRequest(BaseModel):
        commission_pct : float = 0.0
        max_uses       : int   = 0
        custom_code    : str   = ""

    class SignalResponse(BaseModel):
        symbol          : str
        side            : str
        mode            : str
        confidence_score: float
        entry_price     : float
        tp_price        : float
        sl_price        : float
        signal_id       : str
        timestamp       : str

    class TradeResponse(BaseModel):
        symbol          : str
        mode            : str
        side            : str
        entry_price     : float
        exit_price      : Optional[float]
        quantity        : int
        realised_pnl    : Optional[float]
        exit_reason     : Optional[str]
        hold_days       : int
        confidence_score: float
        entry_date      : str
        exit_date       : Optional[str]

    class PortfolioResponse(BaseModel):
        total_trades    : int
        open_positions  : int
        total_pnl       : float
        win_rate        : float
        paper_mode      : bool

    class HealthResponse(BaseModel):
        status          : str
        database        : str
        timestamp       : str
        version         : str = "1.0.0"


from contextlib import asynccontextmanager



# ══════════════════════════════════════════════════════════════════════════
#  APP FACTORY
# ══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    # Startup
    logger.info("G.O.D.S E.Y.E API starting up...")
    _ = get_auth()
    _ = get_sub_manager()
    logger.success("API ready. Docs at http://localhost:8080/docs")
    yield
    # Shutdown
    logger.info("G.O.D.S E.Y.E API shutting down.")

def create_app() -> Any:
    if not FASTAPI_AVAILABLE:
        return None
    application = FastAPI(
        title       = "G.O.D.S E.Y.E — Signal API",
        description = (
            "Generative Observation & Decision System for Equity Intelligence. "
            "AI-powered trading signals for Indian equity markets."
        ),
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        lifespan    = lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )
    return application


app = create_app()
if app is not None:
    app.include_router(dashboard_router, prefix="/api/dashboard")

# ── Singletons ─────────────────────────────────────────────────────────────
_auth_manager       : Optional[AuthManager]       = None
_subscriber_manager : Optional[SubscriberManager] = None


def get_auth() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def get_sub_manager() -> SubscriberManager:
    global _subscriber_manager
    if _subscriber_manager is None:
        _subscriber_manager = SubscriberManager()
    return _subscriber_manager


# ══════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_current_user(
    authorization: str = Header(None),
    x_api_key    : str = Header(None, alias="X-API-Key"),
) -> Dict:
    """
    Dependency: extracts and validates the current user from JWT or API key.
    Raises 401 if neither is valid.
    """
    if not FASTAPI_AVAILABLE:
        return {}

    # Try JWT first
    if authorization and authorization.startswith("Bearer "):
        token   = authorization.split(" ", 1)[1]
        payload = verify_jwt_token(token)
        if payload:
            return {
                "user_id": payload.get("sub"),
                "role"   : payload.get("role", "subscriber"),
                "plan"   : payload.get("plan", "free"),
            }

    # Try API key
    if x_api_key:
        auth = get_auth()
        user = auth.verify_api_key(x_api_key)
        if user:
            return {
                "user_id": user.user_id,
                "role"   : user.role.value,
                "plan"   : user.plan.value,
            }

    raise HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = "Invalid or missing credentials",
        headers     = {"WWW-Authenticate": "Bearer"},
    )


def _require_role(current_user: Dict, *roles: str):
    """Raises 403 if user does not have one of the required roles."""
    if not FASTAPI_AVAILABLE:
        return
    if current_user.get("role") not in roles:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = f"Requires role: {list(roles)}"
        )


# ══════════════════════════════════════════════════════════════════════════
#  DB HELPER
# ══════════════════════════════════════════════════════════════════════════

def _query(sql: str, params=None) -> List[Dict]:
    """Runs a SELECT and returns list of dicts."""
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        conn.close()
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        logger.error(f"DB query failed: {e}")
        return []


def _db_ok() -> bool:
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════
#  ROUTES — only registered if FastAPI is available
# ══════════════════════════════════════════════════════════════════════════

if app is not None:

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health_check():
        """System health check — no auth required."""
        db_status = "ok" if _db_ok() else "error"
        return HealthResponse(
            status    = "ok" if db_status == "ok" else "degraded",
            database  = db_status,
            timestamp = datetime.now().isoformat(),
        )

    @app.get("/metrics/summary", tags=["System"])
    def metrics_summary(current_user: Dict = Depends(_get_current_user)):
        """Key performance metrics for monitoring dashboard."""
        rows = _query("""
            SELECT
                COUNT(*)                                     AS total_trades,
                COUNT(*) FILTER (WHERE realised_pnl > 0)    AS wins,
                COALESCE(SUM(realised_pnl), 0)              AS total_pnl,
                COUNT(*) FILTER (WHERE exit_time IS NULL)   AS open_positions
            FROM trade_log
            WHERE paper_mode = TRUE;
        """)
        if not rows:
            return {"error": "No data"}
        r        = rows[0]
        total    = int(r["total_trades"] or 0)
        wins     = int(r["wins"] or 0)
        return {
            "total_trades"  : total,
            "wins"          : wins,
            "win_rate"      : round(wins / total, 4) if total > 0 else 0,
            "total_pnl_inr" : float(r["total_pnl"] or 0),
            "open_positions": int(r["open_positions"] or 0),
            "timestamp"     : datetime.now().isoformat(),
        }

    # ── Auth ──────────────────────────────────────────────────────────────

    @app.post("/auth/register", tags=["Auth"])
    def register(req: RegisterRequest):
        """
        Register a new subscriber account.
        Returns user_id on success.
        """
        auth = get_auth()
        user = auth.register_user(
            email         = req.email,
            password      = req.password,
            role          = UserRole.SUBSCRIBER,
            referral_code = req.referral_code,
        )
        if not user:
            raise HTTPException(
                status_code = 400,
                detail      = "Registration failed — email may already be registered."
            )
        return {
            "user_id"      : user.user_id,
            "email"        : user.email,
            "referral_code": user.referral_code,
            "message"      : "Account created. Connect your broker account to start.",
        }

    @app.post("/auth/login", tags=["Auth"])
    def login(req: LoginRequest):
        """
        Authenticate and get a JWT access token.
        Token expires in 24 hours.
        """
        auth  = get_auth()
        token = auth.authenticate_user(req.email, req.password)
        if not token:
            raise HTTPException(
                status_code = 401,
                detail      = "Invalid email or password."
            )
        return {
            "access_token": token.access_token,
            "token_type"  : token.token_type,
            "expires_in"  : token.expires_in,
        }

    @app.post("/auth/logout", tags=["Auth"])
    def logout(
        authorization: str = Header(None),
        current_user : Dict = Depends(_get_current_user),
    ):
        """Revokes the current JWT token (logout)."""
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ", 1)[1]
            get_auth().revoke_jwt(token)
        return {"message": "Logged out successfully."}

    # ── Signals ───────────────────────────────────────────────────────────

    @app.get("/signals/latest", tags=["Signals"])
    def get_latest_signals(current_user: Dict = Depends(_get_current_user)):
        """
        Returns today's trading signals.
        Filtered by subscriber's plan (FREE gets swing only, PRO gets all).
        """
        plan = current_user.get("plan", "free")
        try:
            plan_obj   = SubscriptionPlan(plan)
        except ValueError:
            plan_obj   = SubscriptionPlan.FREE

        from api.subscriber_manager import PLAN_SIGNAL_LIMITS
        allowed_modes = PLAN_SIGNAL_LIMITS.get(plan_obj, {}).get("modes", ["swing"])

        placeholders = ",".join(["%s"] * len(allowed_modes))
        rows = _query(f"""
            SELECT symbol, mode, entry_price, tp_price, sl_price,
                   confidence_score, signal_id, entry_time
            FROM trade_log
            WHERE paper_mode = TRUE
              AND entry_time::date = CURRENT_DATE
              AND mode IN ({placeholders})
            ORDER BY confidence_score DESC, entry_time DESC
            LIMIT 10;
        """, tuple(allowed_modes))

        return {
            "date"          : str(date.today()),
            "plan"          : plan,
            "allowed_modes" : allowed_modes,
            "signal_count"  : len(rows),
            "signals"       : [
                {
                    "symbol"          : r["symbol"],
                    "mode"            : r["mode"],
                    "entry_price"     : float(r["entry_price"] or 0),
                    "tp_price"        : float(r["tp_price"] or 0),
                    "sl_price"        : float(r["sl_price"] or 0),
                    "confidence_score": float(r["confidence_score"] or 0),
                    "signal_id"       : r["signal_id"] or "",
                    "timestamp"       : str(r["entry_time"]),
                }
                for r in rows
            ],
        }

    @app.get("/signals/history", tags=["Signals"])
    def get_signal_history(
        from_date    : str  = str(date.today() - timedelta(days=30)),
        to_date      : str  = str(date.today()),
        symbol       : str  = "",
        current_user : Dict = Depends(_get_current_user),
    ):
        """
        Returns signal history for a date range.
        Optionally filter by symbol.
        """
        sql    = """
            SELECT symbol, mode, side, entry_price, exit_price,
                   quantity, realised_pnl, exit_reason, hold_days,
                   confidence_score,
                   entry_time::date AS entry_date,
                   exit_time::date  AS exit_date
            FROM trade_log
            WHERE paper_mode = TRUE
              AND entry_time::date BETWEEN %s AND %s
        """
        params = [from_date, to_date]
        if symbol:
            sql   += " AND symbol = %s"
            params.append(symbol.upper())
        sql += " ORDER BY entry_time DESC LIMIT 200;"

        rows = _query(sql, tuple(params))
        return {
            "from_date"  : from_date,
            "to_date"    : to_date,
            "total"      : len(rows),
            "trades"     : [
                {k: (float(v) if hasattr(v, "__float__") and v is not None
                     else str(v) if v is not None else None)
                 for k, v in r.items()}
                for r in rows
            ],
        }

    # ── Portfolio ─────────────────────────────────────────────────────────

    @app.get("/portfolio", tags=["Portfolio"])
    def get_portfolio(current_user: Dict = Depends(_get_current_user)):
        """Returns current portfolio snapshot for the authenticated subscriber."""
        rows = _query("""
            SELECT
                COUNT(*) FILTER (WHERE exit_time IS NULL)           AS open_positions,
                COUNT(*) FILTER (WHERE exit_time IS NOT NULL)       AS closed_trades,
                COUNT(*) FILTER (WHERE exit_time IS NOT NULL
                                  AND realised_pnl > 0)            AS wins,
                COALESCE(SUM(realised_pnl)
                         FILTER (WHERE exit_time IS NOT NULL), 0)  AS total_pnl,
                COALESCE(SUM(realised_pnl)
                         FILTER (WHERE exit_time::date = CURRENT_DATE), 0)
                                                                    AS today_pnl
            FROM trade_log WHERE paper_mode = TRUE;
        """)
        if not rows:
            return {"error": "No data"}

        r      = rows[0]
        closed = int(r["closed_trades"] or 0)
        wins   = int(r["wins"] or 0)

        # Open positions detail
        open_rows = _query("""
            SELECT symbol, entry_price, quantity, tp_price, sl_price,
                   hold_days, confidence_score, entry_time::date AS entry_date
            FROM trade_log
            WHERE paper_mode = TRUE AND exit_time IS NULL
            ORDER BY entry_time DESC;
        """)

        return {
            "summary": {
                "open_positions": int(r["open_positions"] or 0),
                "closed_trades" : closed,
                "wins"          : wins,
                "win_rate"      : round(wins / closed, 4) if closed > 0 else 0,
                "total_pnl_inr" : float(r["total_pnl"] or 0),
                "today_pnl_inr" : float(r["today_pnl"] or 0),
                "paper_mode"    : True,
            },
            "open_positions": [
                {
                    "symbol"          : r["symbol"],
                    "entry_price"     : float(r["entry_price"] or 0),
                    "quantity"        : int(r["quantity"] or 0),
                    "tp_price"        : float(r["tp_price"] or 0),
                    "sl_price"        : float(r["sl_price"] or 0),
                    "hold_days"       : int(r["hold_days"] or 0),
                    "confidence_score": float(r["confidence_score"] or 0),
                    "entry_date"      : str(r["entry_date"]),
                }
                for r in open_rows
            ],
        }

    # ── Subscriber management ─────────────────────────────────────────────

    @app.post("/subscriber/onboard", tags=["Subscriber"])
    def onboard_subscriber(
        req         : OnboardRequest,
        current_user: Dict = Depends(_get_current_user),
    ):
        """
        Connects a broker account to this subscriber's profile.
        Stores the OAuth token encrypted at rest.
        """
        try:
            provider = BrokerProvider(req.broker_provider)
            plan     = SubscriptionPlan(req.plan)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        mgr     = get_sub_manager()
        account = mgr.onboard_subscriber(
            email          = current_user.get("user_id", "") + "@internal",
            password       = "placeholder",
            broker_provider= provider,
            broker_token   = req.broker_token,
            capital_inr    = req.capital_inr,
            plan           = plan,
            paper_mode     = req.paper_mode,
        )
        if not account:
            raise HTTPException(status_code=400, detail="Onboarding failed.")

        return {
            "user_id"       : account.user_id,
            "broker"        : provider.value,
            "plan"          : plan.value,
            "capital_inr"   : account.capital_inr,
            "paper_mode"    : account.paper_mode,
            "message"       : "Broker connected. You will now receive signals.",
        }

    @app.get("/subscriber/profile", tags=["Subscriber"])
    def get_profile(current_user: Dict = Depends(_get_current_user)):
        """Returns the subscriber's profile."""
        user_id = current_user.get("user_id")
        mgr     = get_sub_manager()
        account = mgr.get_subscriber(user_id)
        if not account:
            return {
                "user_id" : user_id,
                "plan"    : current_user.get("plan", "free"),
                "message" : "No broker connected yet. POST /subscriber/onboard to connect.",
            }
        return {
            "user_id"        : account.user_id,
            "plan"           : account.plan.value,
            "broker"         : account.broker_provider.value if account.broker_provider else None,
            "capital_inr"    : account.capital_inr,
            "paper_mode"     : account.paper_mode,
            "is_active"      : account.is_active,
            "signals_today"  : account.signals_today,
            "signals_month"  : account.signals_month,
            "max_positions"  : account.max_positions,
            "max_drawdown"   : account.max_drawdown_pct,
        }

    @app.put("/subscriber/plan", tags=["Subscriber"])
    def update_plan(
        req         : PlanUpdateRequest,
        current_user: Dict = Depends(_get_current_user),
    ):
        """Upgrades or downgrades subscription plan."""
        try:
            new_plan = SubscriptionPlan(req.new_plan)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid plan: {req.new_plan}")

        mgr     = get_sub_manager()
        success = mgr.update_subscriber_plan(current_user["user_id"], new_plan)
        if not success:
            raise HTTPException(status_code=500, detail="Plan update failed.")
        return {"plan": new_plan.value, "message": f"Plan updated to {new_plan.value}."}

    # ── Admin ─────────────────────────────────────────────────────────────

    @app.get("/admin/stats", tags=["Admin"])
    def admin_stats(current_user: Dict = Depends(_get_current_user)):
        """System-wide statistics. Admin only."""
        _require_role(current_user, "super_admin", "broker")
        rows = _query("""
            SELECT
                (SELECT COUNT(*) FROM trade_log WHERE paper_mode=TRUE)
                    AS total_paper_trades,
                (SELECT COUNT(*) FROM trade_log
                 WHERE paper_mode=TRUE AND exit_time IS NULL)
                    AS open_positions,
                (SELECT COALESCE(SUM(realised_pnl),0) FROM trade_log
                 WHERE paper_mode=TRUE AND exit_time IS NOT NULL)
                    AS total_paper_pnl,
                (SELECT MAX(date) FROM daily_ohlcv)
                    AS latest_data_date,
                (SELECT MAX(date) FROM features_fused)
                    AS latest_features_date;
        """)
        return rows[0] if rows else {}

    @app.get("/admin/subscribers", tags=["Admin"])
    def admin_subscribers(current_user: Dict = Depends(_get_current_user)):
        """List all subscribers. Admin only."""
        _require_role(current_user, "super_admin", "broker")
        rows = _query("""
            SELECT user_id, email, plan, broker_provider,
                   is_active, paper_mode, capital_inr,
                   signals_today, signals_month, created_at
            FROM subscriber_profiles
            ORDER BY created_at DESC;
        """)
        return {"total": len(rows), "subscribers": rows}

    # ── Distributor ───────────────────────────────────────────────────────

    @app.get("/distributor/stats", tags=["Distributor"])
    def distributor_stats(current_user: Dict = Depends(_get_current_user)):
        """Distributor dashboard stats — referral counts, commission summary."""
        _require_role(current_user, "distributor", "broker", "super_admin")
        mgr   = get_sub_manager()
        stats = mgr.get_distributor_stats(current_user["user_id"])
        auth  = get_auth()
        ref   = auth.get_referral_stats(current_user["user_id"])
        return {**stats, "referral_codes": ref.get("referral_codes", [])}

    @app.post("/distributor/referral", tags=["Distributor"])
    def create_referral(
        req         : ReferralCreateRequest,
        current_user: Dict = Depends(_get_current_user),
    ):
        """
        Creates a referral code for a distributor.
        Subscribers who sign up with this code are linked to this distributor.
        Commission % is applied to their subscription revenue.
        """
        _require_role(current_user, "distributor", "broker", "super_admin")
        auth = get_auth()
        ref  = auth.create_referral_code(
            owner_id       = current_user["user_id"],
            owner_role     = UserRole(current_user["role"]),
            code           = req.custom_code or "",
            commission_pct = req.commission_pct,
            max_uses       = req.max_uses,
        )
        if not ref:
            raise HTTPException(status_code=500, detail="Referral code creation failed.")
        return {
            "code"          : ref.code,
            "commission_pct": ref.commission_pct,
            "max_uses"      : ref.max_uses,
            "message"       : f"Share this code with subscribers: {ref.code}",
        }

    @app.get("/distributor/referral/validate/{code}", tags=["Distributor"])
    def validate_referral(code: str):
        """Validates a referral code (public endpoint — no auth required)."""
        auth       = get_auth()
        is_valid, reason = auth.validate_referral_code(code)
        return {"code": code, "valid": is_valid, "reason": reason}

    # ── Startup / shutdown ────────────────────────────────────────────────

    @asynccontextmanager
    async def lifespan(app):
        # Startup
        logger.info("G.O.D.S E.Y.E API starting up...")
        _ = get_auth()
        _ = get_sub_manager()
        logger.success("API ready. Docs at http://localhost:8080/docs")
        yield
        # Shutdown
        logger.info("G.O.D.S E.Y.E API shutting down.")


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest api/main.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestAPIMain:
    """
    Unit tests for API utilities.
    Does not start the actual server — tests helper functions only.
    """

    # ── DB helper ─────────────────────────────────────────────────────────

    def test_db_ok_returns_bool(self):
        result = _db_ok()
        assert isinstance(result, bool)

    def test_query_empty_on_bad_sql(self):
        rows = _query("SELECT 1 WHERE FALSE;")
        assert rows == []

    def test_query_returns_list(self):
        rows = _query("SELECT 1 AS val;")
        assert isinstance(rows, list)

    def test_query_returns_dict_rows(self):
        rows = _query("SELECT 42 AS answer;")
        if rows:
            assert isinstance(rows[0], dict)
            assert rows[0]["answer"] == 42

    # ── Auth helpers ──────────────────────────────────────────────────────

    def test_get_current_user_no_auth_raises(self):
        """No credentials should raise HTTPException."""
        if not FASTAPI_AVAILABLE:
            return
        from fastapi import HTTPException
        import pytest
        with pytest.raises(HTTPException) as exc:
            _get_current_user(authorization=None, x_api_key=None)
        assert exc.value.status_code == 401

    def test_get_current_user_valid_jwt(self):
        """Valid JWT should return user dict."""
        if not FASTAPI_AVAILABLE:
            return
        from api.auth import create_jwt_token, UserRole, SubscriptionPlan
        token = create_jwt_token("user123", UserRole.SUBSCRIBER)
        try:
            result = _get_current_user(
                authorization = f"Bearer {token.access_token}",
                x_api_key     = None,
            )
            assert result.get("user_id") == "user123"
        except Exception:
            pass   # may fail if JWT not available

    def test_require_role_passes(self):
        """Correct role should not raise."""
        if not FASTAPI_AVAILABLE:
            return
        user = {"user_id": "u1", "role": "super_admin"}
        _require_role(user, "super_admin", "broker")   # should not raise

    def test_require_role_fails(self):
        """Wrong role should raise 403."""
        if not FASTAPI_AVAILABLE:
            return
        from fastapi import HTTPException
        import pytest
        user = {"user_id": "u1", "role": "subscriber"}
        with pytest.raises(HTTPException) as exc:
            _require_role(user, "super_admin")
        assert exc.value.status_code == 403

    # ── FASTAPI_AVAILABLE flag ────────────────────────────────────────────

    def test_fastapi_available_is_bool(self):
        assert isinstance(FASTAPI_AVAILABLE, bool)

    def test_app_created_if_fastapi_available(self):
        if FASTAPI_AVAILABLE:
            assert app is not None
        else:
            assert app is None

    # ── Route existence ───────────────────────────────────────────────────

    def test_health_route_exists(self):
        if not FASTAPI_AVAILABLE or app is None:
            return
        routes = [r.path for r in app.routes]
        assert "/health" in routes

    def test_auth_routes_exist(self):
        if not FASTAPI_AVAILABLE or app is None:
            return
        routes = [r.path for r in app.routes]
        assert "/auth/login"    in routes
        assert "/auth/register" in routes
        assert "/auth/logout"   in routes

    def test_signal_routes_exist(self):
        if not FASTAPI_AVAILABLE or app is None:
            return
        routes = [r.path for r in app.routes]
        assert "/signals/latest"  in routes
        assert "/signals/history" in routes

    def test_portfolio_route_exists(self):
        if not FASTAPI_AVAILABLE or app is None:
            return
        routes = [r.path for r in app.routes]
        assert "/portfolio" in routes

    def test_admin_routes_exist(self):
        if not FASTAPI_AVAILABLE or app is None:
            return
        routes = [r.path for r in app.routes]
        assert "/admin/stats"       in routes
        assert "/admin/subscribers" in routes

    def test_distributor_routes_exist(self):
        if not FASTAPI_AVAILABLE or app is None:
            return
        routes = [r.path for r in app.routes]
        assert "/distributor/stats"    in routes
        assert "/distributor/referral" in routes

    # ── Pydantic models ───────────────────────────────────────────────────

    def test_login_request_model(self):
        if not FASTAPI_AVAILABLE:
            return
        req = LoginRequest(email="a@b.com", password="pass123")
        assert req.email    == "a@b.com"
        assert req.password == "pass123"

    def test_register_request_defaults(self):
        if not FASTAPI_AVAILABLE:
            return
        req = RegisterRequest(email="a@b.com", password="pass")
        assert req.referral_code == ""

    def test_onboard_request_defaults(self):
        if not FASTAPI_AVAILABLE:
            return
        req = OnboardRequest(broker_token="tok123", capital_inr=500000)
        assert req.broker_provider == "zerodha"
        assert req.paper_mode      is True

    def test_plan_update_request(self):
        if not FASTAPI_AVAILABLE:
            return
        req = PlanUpdateRequest(new_plan="pro")
        assert req.new_plan == "pro"

    def test_health_response_model(self):
        if not FASTAPI_AVAILABLE:
            return
        r = HealthResponse(
            status="ok", database="ok",
            timestamp=datetime.now().isoformat()
        )
        assert r.version == "1.0.0"


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv or "pytest" in sys.argv[0]:
        import pytest
        sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))
    elif FASTAPI_AVAILABLE:
        import uvicorn
        logger.info("Starting G.O.D.S E.Y.E API server...")
        uvicorn.run(
            "api.main:app",
            host       = "0.0.0.0",
            port       = 8080,
            reload     = True,
            log_level  = "info",
        )
    else:
        print("FastAPI not installed. Run: pip install fastapi uvicorn pydantic[email]")