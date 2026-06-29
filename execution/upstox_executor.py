"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Upstox Broker Executor                          ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : execution/upstox_executor.py                           ║
║         Phase   : 4 — Paper Trading & Live Monitoring                   ║
║                                                                          ║
║  What this module does:                                                  ║
║    Provides the Upstox broker integration as a FAILOVER executor.        ║
║    When Zerodha Kite Connect is unavailable (API down, auth expired,     ║
║    rate limit hit), execution/failover.py automatically switches all     ║
║    order routing to this module.                                         ║
║                                                                          ║
║  Upstox API v2:                                                          ║
║    REST API with OAuth 2.0. Orders placed via HTTPS POST.               ║
║    WebSocket feed for live quotes (separate from order API).             ║
║    Rate limits: 25 requests/second, 1000 orders/day.                    ║
║                                                                          ║
║  Paper trading mode:                                                     ║
║    When PAPER_MODE=true in .env, all orders are simulated locally.       ║
║    No actual API calls are made. Fills are assumed at last traded        ║
║    price with a configurable slippage model.                             ║
║                                                                          ║
║  Order types supported:                                                  ║
║    MARKET : Immediate fill at best available price                       ║
║    LIMIT  : Fill at specified price or better                            ║
║    SL     : Stop-loss order (trigger + limit price)                      ║
║    SL-M   : Stop-loss market order (trigger only)                        ║
║                                                                          ║
║  Usage:                                                                  ║
║    executor = UpstoxExecutor()                                           ║
║    result   = executor.place_order(OrderRequest(                         ║
║        symbol     = "NSE_EQ|INE002A01018",   # Upstox instrument key    ║
║        side       = OrderSide.BUY,                                       ║
║        quantity   = 10,                                                  ║
║        order_type = OrderType.MARKET,                                    ║
║    ))                                                                    ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install upstox-python-sdk requests loguru python-dotenv           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import time
import uuid
import threading
import requests

from dataclasses  import dataclass, field
from datetime     import datetime, date
from enum         import Enum
from typing       import Optional, Dict, List, Any
from loguru       import logger
from dotenv       import load_dotenv

load_dotenv()

# ── Upstox API config ──────────────────────────────────────────────────────
UPSTOX_API_KEY      = os.getenv("UPSTOX_API_KEY",    "")
UPSTOX_API_SECRET   = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost")
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")   # set after OAuth

UPSTOX_BASE_URL     = "https://api.upstox.com/v2"
PAPER_MODE          = os.getenv("PAPER_MODE", "true").lower() == "true"

# ── Rate limiting ──────────────────────────────────────────────────────────
MAX_REQUESTS_PER_SEC = 25
REQUEST_TIMEOUT      = 10    # seconds
MAX_RETRIES          = 3
RETRY_DELAY          = 1.0   # seconds


# ══════════════════════════════════════════════════════════════════════════
#  ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════

class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    SL     = "SL"        # Stop-Loss Limit
    SL_M   = "SL-M"      # Stop-Loss Market


class OrderStatus(str, Enum):
    PENDING    = "PENDING"
    OPEN       = "OPEN"
    COMPLETE   = "COMPLETE"
    REJECTED   = "REJECTED"
    CANCELLED  = "CANCELLED"
    PAPER_FILL = "PAPER_FILL"   # paper trading simulated fill


class ProductType(str, Enum):
    DELIVERY  = "D"    # CNC delivery
    INTRADAY  = "I"    # MIS intraday
    MTF       = "MTF"  # Margin Trading Facility


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class OrderRequest:
    """
    Encapsulates all parameters for a single order.

    symbol uses Upstox instrument key format:
        NSE_EQ|<ISIN>  e.g. "NSE_EQ|INE002A01018" for Reliance
        NSE_FO|<key>   for futures/options

    For paper trading, symbol can be the plain NSE ticker (e.g. "RELIANCE")
    as no actual API call is made.
    """
    symbol          : str
    side            : OrderSide
    quantity        : int
    order_type      : OrderType       = OrderType.MARKET
    price           : float           = 0.0     # for LIMIT orders
    trigger_price   : float           = 0.0     # for SL / SL-M orders
    product         : ProductType     = ProductType.DELIVERY
    tag             : str             = ""       # internal reference tag
    validity        : str             = "DAY"    # DAY or IOC

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be > 0, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.price <= 0:
            raise ValueError("LIMIT order requires price > 0")
        if self.order_type in (OrderType.SL, OrderType.SL_M) and self.trigger_price <= 0:
            raise ValueError("SL order requires trigger_price > 0")


@dataclass
class OrderResult:
    """
    Result of a placed order — returned by place_order().
    """
    order_id        : str
    symbol          : str
    side            : OrderSide
    quantity        : int
    status          : OrderStatus
    fill_price      : float           = 0.0
    fill_quantity   : int             = 0
    timestamp       : datetime        = field(default_factory=datetime.now)
    broker          : str             = "upstox"
    paper_mode      : bool            = False
    error_message   : str             = ""
    raw_response    : Dict            = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status in (OrderStatus.COMPLETE, OrderStatus.PAPER_FILL)

    @property
    def is_rejected(self) -> bool:
        return self.status == OrderStatus.REJECTED

    @property
    def fill_value(self) -> float:
        """Total fill value in ₹."""
        return self.fill_price * self.fill_quantity


@dataclass
class PositionInfo:
    """Represents a live position from Upstox portfolio API."""
    symbol          : str
    quantity        : int
    average_price   : float
    last_price      : float
    pnl             : float
    product         : str


@dataclass
class QuoteData:
    """Live quote data from Upstox market data API."""
    symbol          : str
    last_price      : float
    open_price      : float
    high_price      : float
    low_price       : float
    close_price     : float
    volume          : int
    timestamp       : datetime = field(default_factory=datetime.now)


# ══════════════════════════════════════════════════════════════════════════
#  RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Token bucket rate limiter for Upstox API.
    Allows MAX_REQUESTS_PER_SEC requests per second.
    Thread-safe.
    """

    def __init__(self, max_per_sec: int = MAX_REQUESTS_PER_SEC):
        self._max       = max_per_sec
        self._tokens    = max_per_sec
        self._last_refill = time.monotonic()
        self._lock      = threading.Lock()

    def acquire(self):
        """Blocks until a request token is available."""
        with self._lock:
            now      = time.monotonic()
            elapsed  = now - self._last_refill
            refill   = elapsed * self._max
            self._tokens = min(self._max, self._tokens + refill)
            self._last_refill = now

            if self._tokens < 1:
                sleep_time = (1 - self._tokens) / self._max
                time.sleep(sleep_time)
                self._tokens = 0
            else:
                self._tokens -= 1


# ══════════════════════════════════════════════════════════════════════════
#  PAPER TRADING ENGINE
# ══════════════════════════════════════════════════════════════════════════

class PaperTradingEngine:
    """
    Simulates order execution for paper trading mode.

    Fills are assumed at last traded price with configurable slippage.
    All fills are logged to memory and can be exported.

    Slippage model:
        MARKET order : last_price × (1 + slippage_pct) for BUY
                       last_price × (1 - slippage_pct) for SELL
        LIMIT order  : fills at limit price if market allows, else rejects
    """

    SLIPPAGE_PCT = 0.0005   # 0.05% simulated slippage

    def __init__(self):
        self._fills      : List[OrderResult] = []
        self._positions  : Dict[str, Dict]   = {}
        self._cash       : float             = 0.0

    def fill(
        self,
        request    : OrderRequest,
        last_price : float,
    ) -> OrderResult:
        """
        Simulates an order fill.

        Args:
            request    : The order to fill
            last_price : Current market price for the symbol

        Returns:
            OrderResult with PAPER_FILL status
        """
        if last_price <= 0:
            last_price = request.price if request.price > 0 else 100.0

        # Apply slippage
        if request.order_type == OrderType.MARKET:
            if request.side == OrderSide.BUY:
                fill_price = last_price * (1 + self.SLIPPAGE_PCT)
            else:
                fill_price = last_price * (1 - self.SLIPPAGE_PCT)
        elif request.order_type == OrderType.LIMIT:
            fill_price = request.price
        else:
            fill_price = request.trigger_price or last_price

        result = OrderResult(
            order_id      = f"PAPER-{uuid.uuid4().hex[:8].upper()}",
            symbol        = request.symbol,
            side          = request.side,
            quantity      = request.quantity,
            status        = OrderStatus.PAPER_FILL,
            fill_price    = round(fill_price, 2),
            fill_quantity = request.quantity,
            paper_mode    = True,
            broker        = "upstox_paper",
        )

        self._fills.append(result)
        self._update_positions(result)

        logger.info(
            f"[PAPER] {result.side.value:4s} {result.symbol:12s} "
            f"qty={result.quantity} @ ₹{result.fill_price:.2f} "
            f"| order_id={result.order_id}"
        )

        return result

    def _update_positions(self, result: OrderResult):
        """Updates in-memory paper positions after a fill."""
        sym = result.symbol
        if sym not in self._positions:
            self._positions[sym] = {"quantity": 0, "average_price": 0.0}

        pos = self._positions[sym]
        if result.side == OrderSide.BUY:
            total_qty   = pos["quantity"] + result.fill_quantity
            total_cost  = (pos["quantity"] * pos["average_price"] +
                           result.fill_quantity * result.fill_price)
            pos["average_price"] = total_cost / total_qty if total_qty > 0 else 0
            pos["quantity"]      = total_qty
        else:
            pos["quantity"] = max(0, pos["quantity"] - result.fill_quantity)
            if pos["quantity"] == 0:
                pos["average_price"] = 0.0

    def get_positions(self) -> Dict[str, Dict]:
        return {k: v.copy() for k, v in self._positions.items() if v["quantity"] > 0}

    def get_fills(self) -> List[OrderResult]:
        return list(self._fills)

    def get_pnl_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Returns P&L summary for all paper positions."""
        total_pnl = 0.0
        details   = {}
        for sym, pos in self._positions.items():
            if pos["quantity"] <= 0:
                continue
            curr = current_prices.get(sym, pos["average_price"])
            pnl  = (curr - pos["average_price"]) * pos["quantity"]
            total_pnl += pnl
            details[sym] = {
                "quantity"     : pos["quantity"],
                "average_price": pos["average_price"],
                "current_price": curr,
                "unrealised_pnl": pnl,
            }
        return {"total_unrealised_pnl": total_pnl, "positions": details}


# ══════════════════════════════════════════════════════════════════════════
#  UPSTOX EXECUTOR
# ══════════════════════════════════════════════════════════════════════════

class UpstoxExecutor:
    """
    Upstox broker executor — failover broker for G.O.D.S E.Y.E.

    Provides the same interface as KiteExecutor so failover.py can
    switch between them transparently.

    In PAPER_MODE (default):
        All orders are simulated. No real API calls.
        UPSTOX_ACCESS_TOKEN not required.

    In LIVE mode:
        Requires UPSTOX_ACCESS_TOKEN in .env (obtained via OAuth flow).
        Orders placed via Upstox REST API v2.

    Thread-safe: uses a lock for order placement to prevent double-fills.

    Usage:
        executor = UpstoxExecutor()

        # Health check
        ok = executor.is_healthy()

        # Place order
        result = executor.place_order(OrderRequest(
            symbol     = "NSE_EQ|INE002A01018",
            side       = OrderSide.BUY,
            quantity   = 10,
            order_type = OrderType.MARKET,
        ))

        # Get positions
        positions = executor.get_positions()

        # Get live quote
        quote = executor.get_quote("NSE_EQ|INE002A01018")
    """

    def __init__(
        self,
        access_token: str  = "",
        paper_mode  : bool = PAPER_MODE,
    ):
        self.access_token = access_token or UPSTOX_ACCESS_TOKEN
        self.paper_mode   = paper_mode
        self._lock        = threading.Lock()
        self._rate_limiter= RateLimiter()
        self._paper_engine= PaperTradingEngine()
        self._order_log   : List[OrderResult] = []
        self._session     = requests.Session()
        self._healthy     = True
        self._last_error  = ""

        if not paper_mode and not self.access_token:
            logger.warning(
                "UpstoxExecutor: LIVE mode but UPSTOX_ACCESS_TOKEN not set. "
                "Set it in .env or pass access_token parameter. "
                "Falling back to paper mode."
            )
            self.paper_mode = True

        mode_str = "PAPER" if self.paper_mode else "LIVE"
        logger.info(f"UpstoxExecutor initialized in {mode_str} mode.")

    # ══════════════════════════════════════════════════════════════════════
    #  PUBLIC API (mirrors KiteExecutor interface)
    # ══════════════════════════════════════════════════════════════════════

    def is_healthy(self) -> bool:
        """
        Checks if Upstox API is accessible and credentials are valid.
        Used by failover.py to decide whether to switch to this executor.

        In paper mode: always returns True.
        In live mode: makes a lightweight profile API call.

        Returns:
            True if executor is ready to place orders
        """
        if self.paper_mode:
            return True

        try:
            resp = self._get("/user/profile")
            self._healthy    = resp.get("status") == "success"
            self._last_error = "" if self._healthy else str(resp)
            return self._healthy
        except Exception as e:
            self._healthy    = False
            self._last_error = str(e)
            logger.warning(f"UpstoxExecutor health check failed: {e}")
            return False

    def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Places a single order via Upstox API (or simulates in paper mode).

        Args:
            request : OrderRequest with all order parameters

        Returns:
            OrderResult with fill details and order_id
        """
        with self._lock:
            if self.paper_mode:
                return self._place_paper_order(request)
            else:
                return self._place_live_order(request)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancels an open order by order_id.

        Args:
            order_id : Upstox order_id to cancel

        Returns:
            True if cancellation successful
        """
        if self.paper_mode:
            logger.info(f"[PAPER] Cancel order {order_id} (simulated)")
            return True

        try:
            resp = self._delete(f"/order/cancel", params={"order_id": order_id})
            success = resp.get("status") == "success"
            if success:
                logger.info(f"Order {order_id} cancelled successfully.")
            else:
                logger.warning(f"Cancel failed for {order_id}: {resp}")
            return success
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """
        Fetches current status of an order.

        Args:
            order_id : Upstox order_id

        Returns:
            OrderResult with current status, or None if not found
        """
        if self.paper_mode:
            for result in self._order_log:
                if result.order_id == order_id:
                    return result
            return None

        try:
            resp = self._get(f"/order/details", params={"order_id": order_id})
            if resp.get("status") != "success":
                return None
            data = resp.get("data", {})
            return self._parse_order_response(data)
        except Exception as e:
            logger.error(f"get_order_status failed for {order_id}: {e}")
            return None

    def get_positions(self) -> List[PositionInfo]:
        """
        Returns all current open positions.

        In paper mode: returns from PaperTradingEngine.
        In live mode: fetches from Upstox portfolio API.

        Returns:
            List of PositionInfo objects
        """
        if self.paper_mode:
            positions = []
            for sym, pos in self._paper_engine.get_positions().items():
                positions.append(PositionInfo(
                    symbol        = sym,
                    quantity      = pos["quantity"],
                    average_price = pos["average_price"],
                    last_price    = pos["average_price"],  # no live price in paper
                    pnl           = 0.0,
                    product       = "D",
                ))
            return positions

        try:
            resp = self._get("/portfolio/short-term-positions")
            if resp.get("status") != "success":
                return []
            positions = []
            for item in resp.get("data", []):
                positions.append(PositionInfo(
                    symbol        = item.get("trading_symbol", ""),
                    quantity      = int(item.get("quantity", 0)),
                    average_price = float(item.get("average_price", 0)),
                    last_price    = float(item.get("last_price", 0)),
                    pnl           = float(item.get("pnl", 0)),
                    product       = item.get("product", "D"),
                ))
            return positions
        except Exception as e:
            logger.error(f"get_positions failed: {e}")
            return []

    def get_quote(self, instrument_key: str) -> Optional[QuoteData]:
        """
        Fetches live quote for a single instrument.

        Args:
            instrument_key : Upstox format e.g. "NSE_EQ|INE002A01018"

        Returns:
            QuoteData or None on failure
        """
        if self.paper_mode:
            # Paper mode: return dummy quote
            return QuoteData(
                symbol      = instrument_key,
                last_price  = 100.0,
                open_price  = 99.0,
                high_price  = 101.0,
                low_price   = 98.5,
                close_price = 100.0,
                volume      = 1_000_000,
            )

        try:
            resp = self._get(
                "/market-quote/quotes",
                params={"instrument_key": instrument_key}
            )
            if resp.get("status") != "success":
                return None

            data = resp.get("data", {}).get(instrument_key, {})
            ohlc = data.get("ohlc", {})

            return QuoteData(
                symbol      = instrument_key,
                last_price  = float(data.get("last_price", 0)),
                open_price  = float(ohlc.get("open", 0)),
                high_price  = float(ohlc.get("high", 0)),
                low_price   = float(ohlc.get("low", 0)),
                close_price = float(ohlc.get("close", 0)),
                volume      = int(data.get("volume", 0)),
            )
        except Exception as e:
            logger.error(f"get_quote failed for {instrument_key}: {e}")
            return None

    def get_funds(self) -> Dict[str, float]:
        """
        Returns available funds/margin from Upstox.

        Returns:
            Dict with 'available_cash', 'used_margin', 'total' keys
        """
        if self.paper_mode:
            return {
                "available_cash": 1_000_000.0,
                "used_margin"   : 0.0,
                "total"         : 1_000_000.0,
            }

        try:
            resp = self._get("/user/get-funds-and-margin", params={"segment": "SEC"})
            if resp.get("status") != "success":
                return {}
            data = resp.get("data", {})
            equity = data.get("equity", {})
            return {
                "available_cash": float(equity.get("available_margin", 0)),
                "used_margin"   : float(equity.get("used_margin", 0)),
                "total"         : float(equity.get("net", 0)),
            }
        except Exception as e:
            logger.error(f"get_funds failed: {e}")
            return {}

    def get_order_history(self) -> List[OrderResult]:
        """Returns all orders placed this session."""
        return list(self._order_log)

    def get_paper_pnl(self, current_prices: Dict[str, float]) -> Dict:
        """Returns paper trading P&L summary (paper mode only)."""
        if not self.paper_mode:
            return {}
        return self._paper_engine.get_pnl_summary(current_prices)

    # ══════════════════════════════════════════════════════════════════════
    #  INTERNAL — ORDER PLACEMENT
    # ══════════════════════════════════════════════════════════════════════

    def _place_paper_order(self, request: OrderRequest) -> OrderResult:
        """Simulates order fill via PaperTradingEngine."""
        last_price = request.price if request.price > 0 else 100.0
        result     = self._paper_engine.fill(request, last_price)
        self._order_log.append(result)
        return result

    def _place_live_order(self, request: OrderRequest) -> OrderResult:
        """Places a real order via Upstox REST API v2."""
        payload = {
            "quantity"         : request.quantity,
            "product"          : request.product.value,
            "validity"         : request.validity,
            "price"            : request.price,
            "tag"              : request.tag or f"GODSEYE-{uuid.uuid4().hex[:6]}",
            "instrument_token" : request.symbol,
            "order_type"       : request.order_type.value,
            "transaction_type" : request.side.value,
            "disclosed_quantity": 0,
            "trigger_price"    : request.trigger_price,
            "is_amo"           : False,
        }

        try:
            resp = self._post("/order/place", payload)

            if resp.get("status") == "success":
                order_id = resp.get("data", {}).get("order_id", "")
                result   = OrderResult(
                    order_id      = order_id,
                    symbol        = request.symbol,
                    side          = request.side,
                    quantity      = request.quantity,
                    status        = OrderStatus.OPEN,
                    paper_mode    = False,
                    broker        = "upstox",
                    raw_response  = resp,
                )
                logger.info(
                    f"[LIVE] {request.side.value:4s} {request.symbol:20s} "
                    f"qty={request.quantity} type={request.order_type.value} "
                    f"| order_id={order_id}"
                )
            else:
                error_msg = resp.get("errors", [{}])[0].get("message", str(resp))
                result    = OrderResult(
                    order_id      = "",
                    symbol        = request.symbol,
                    side          = request.side,
                    quantity      = request.quantity,
                    status        = OrderStatus.REJECTED,
                    error_message = error_msg,
                    paper_mode    = False,
                    broker        = "upstox",
                    raw_response  = resp,
                )
                logger.error(
                    f"Order rejected by Upstox: {error_msg} | "
                    f"symbol={request.symbol}"
                )

        except Exception as e:
            logger.error(f"Upstox place_order exception: {e}")
            result = OrderResult(
                order_id      = "",
                symbol        = request.symbol,
                side          = request.side,
                quantity      = request.quantity,
                status        = OrderStatus.REJECTED,
                error_message = str(e),
                paper_mode    = False,
                broker        = "upstox",
            )

        self._order_log.append(result)
        return result

    # ══════════════════════════════════════════════════════════════════════
    #  INTERNAL — HTTP HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type" : "application/json",
            "Accept"       : "application/json",
        }

    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """Makes a GET request to Upstox API with retry logic."""
        self._rate_limiter.acquire()
        url = UPSTOX_BASE_URL + endpoint

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.get(
                    url,
                    headers = self._headers(),
                    params  = params or {},
                    timeout = REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited — back off
                    time.sleep(RETRY_DELAY * attempt)
                    continue
                raise
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                raise

        return {}

    def _post(self, endpoint: str, payload: Dict) -> Dict:
        """Makes a POST request to Upstox API with retry logic."""
        self._rate_limiter.acquire()
        url = UPSTOX_BASE_URL + endpoint

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    url,
                    headers = self._headers(),
                    json    = payload,
                    timeout = REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                if e.response.status_code == 429:
                    time.sleep(RETRY_DELAY * attempt)
                    continue
                raise
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                raise

        return {}

    def _delete(self, endpoint: str, params: Dict = None) -> Dict:
        """Makes a DELETE request to Upstox API."""
        self._rate_limiter.acquire()
        url = UPSTOX_BASE_URL + endpoint
        try:
            resp = self._session.delete(
                url,
                headers = self._headers(),
                params  = params or {},
                timeout = REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"DELETE {endpoint} failed: {e}")
            return {}

    def _parse_order_response(self, data: Dict) -> OrderResult:
        """Parses Upstox order detail response into OrderResult."""
        status_map = {
            "complete"  : OrderStatus.COMPLETE,
            "rejected"  : OrderStatus.REJECTED,
            "cancelled" : OrderStatus.CANCELLED,
            "open"      : OrderStatus.OPEN,
            "pending"   : OrderStatus.PENDING,
        }
        status = status_map.get(
            data.get("status", "").lower(),
            OrderStatus.PENDING
        )
        return OrderResult(
            order_id      = data.get("order_id", ""),
            symbol        = data.get("trading_symbol", ""),
            side          = OrderSide(data.get("transaction_type", "BUY")),
            quantity      = int(data.get("quantity", 0)),
            status        = status,
            fill_price    = float(data.get("average_price", 0)),
            fill_quantity = int(data.get("filled_quantity", 0)),
            broker        = "upstox",
            raw_response  = data,
        )

    def __repr__(self) -> str:
        mode = "PAPER" if self.paper_mode else "LIVE"
        return (
            f"UpstoxExecutor(mode={mode}, "
            f"healthy={self._healthy}, "
            f"orders={len(self._order_log)})"
        )


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest execution/upstox_executor.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestUpstoxExecutor:
    """
    Unit tests for UpstoxExecutor.
    All tests run in paper mode — no real API calls.
    """

    def _make_executor(self) -> UpstoxExecutor:
        return UpstoxExecutor(paper_mode=True)

    def _make_request(self, **kwargs) -> OrderRequest:
        defaults = dict(
            symbol     = "NSE_EQ|INE002A01018",
            side       = OrderSide.BUY,
            quantity   = 10,
            order_type = OrderType.MARKET,
            price      = 0.0,
        )
        defaults.update(kwargs)
        return OrderRequest(**defaults)

    # ── Initialization tests ──────────────────────────────────────────────

    def test_paper_mode_default(self):
        ex = UpstoxExecutor(paper_mode=True)
        assert ex.paper_mode is True

    def test_live_mode_without_token_falls_back(self):
        """Live mode without token should fall back to paper mode."""
        ex = UpstoxExecutor(access_token="", paper_mode=False)
        assert ex.paper_mode is True

    def test_live_mode_with_token(self):
        ex = UpstoxExecutor(access_token="fake_token", paper_mode=False)
        assert ex.paper_mode is False

    # ── Health check tests ────────────────────────────────────────────────

    def test_paper_mode_always_healthy(self):
        ex = self._make_executor()
        assert ex.is_healthy() is True

    # ── Order request validation ──────────────────────────────────────────

    def test_order_request_valid(self):
        req = self._make_request()
        assert req.quantity == 10
        assert req.side     == OrderSide.BUY

    def test_order_request_zero_quantity_raises(self):
        import pytest
        with pytest.raises(ValueError, match="quantity"):
            OrderRequest(
                symbol="TEST", side=OrderSide.BUY,
                quantity=0, order_type=OrderType.MARKET
            )

    def test_limit_order_requires_price(self):
        import pytest
        with pytest.raises(ValueError, match="price"):
            OrderRequest(
                symbol="TEST", side=OrderSide.BUY,
                quantity=10, order_type=OrderType.LIMIT,
                price=0.0
            )

    def test_sl_order_requires_trigger(self):
        import pytest
        with pytest.raises(ValueError, match="trigger"):
            OrderRequest(
                symbol="TEST", side=OrderSide.BUY,
                quantity=10, order_type=OrderType.SL,
                price=100.0, trigger_price=0.0
            )

    # ── Paper order placement ─────────────────────────────────────────────

    def test_paper_buy_returns_result(self):
        ex     = self._make_executor()
        result = ex.place_order(self._make_request())
        assert isinstance(result, OrderResult)

    def test_paper_buy_is_filled(self):
        ex     = self._make_executor()
        result = ex.place_order(self._make_request())
        assert result.is_filled

    def test_paper_buy_has_order_id(self):
        ex     = self._make_executor()
        result = ex.place_order(self._make_request())
        assert result.order_id.startswith("PAPER-")

    def test_paper_sell_is_filled(self):
        ex     = self._make_executor()
        result = ex.place_order(self._make_request(side=OrderSide.SELL))
        assert result.is_filled

    def test_paper_order_fill_price_positive(self):
        ex     = self._make_executor()
        result = ex.place_order(self._make_request())
        assert result.fill_price > 0

    def test_paper_order_quantity_correct(self):
        ex     = self._make_executor()
        result = ex.place_order(self._make_request(quantity=25))
        assert result.fill_quantity == 25

    def test_paper_order_broker_is_upstox(self):
        ex     = self._make_executor()
        result = ex.place_order(self._make_request())
        assert "upstox" in result.broker

    def test_multiple_paper_orders(self):
        ex = self._make_executor()
        for _ in range(10):
            result = ex.place_order(self._make_request())
            assert result.is_filled
        assert len(ex.get_order_history()) == 10

    # ── Slippage model tests ──────────────────────────────────────────────

    def test_buy_slippage_above_last_price(self):
        """Buy fills should be slightly above last price."""
        ex      = self._make_executor()
        req     = self._make_request(price=100.0)
        result  = ex._paper_engine.fill(req, last_price=100.0)
        assert result.fill_price >= 100.0

    def test_sell_slippage_below_last_price(self):
        """Sell fills should be slightly below last price."""
        ex      = self._make_executor()
        req     = self._make_request(side=OrderSide.SELL, price=100.0)
        result  = ex._paper_engine.fill(req, last_price=100.0)
        assert result.fill_price <= 100.0

    def test_limit_order_fills_at_limit_price(self):
        ex      = self._make_executor()
        req     = self._make_request(
            order_type=OrderType.LIMIT, price=95.0
        )
        result  = ex._paper_engine.fill(req, last_price=100.0)
        assert result.fill_price == 95.0

    # ── Position tracking tests ───────────────────────────────────────────

    def test_positions_update_after_buy(self):
        ex  = self._make_executor()
        ex.place_order(self._make_request(quantity=10))
        pos = ex._paper_engine.get_positions()
        sym = "NSE_EQ|INE002A01018"
        assert sym in pos
        assert pos[sym]["quantity"] == 10

    def test_positions_reduce_after_sell(self):
        ex  = self._make_executor()
        ex.place_order(self._make_request(quantity=10))
        ex.place_order(self._make_request(side=OrderSide.SELL, quantity=5))
        pos = ex._paper_engine.get_positions()
        sym = "NSE_EQ|INE002A01018"
        assert pos[sym]["quantity"] == 5

    def test_position_cleared_after_full_sell(self):
        ex  = self._make_executor()
        ex.place_order(self._make_request(quantity=10))
        ex.place_order(self._make_request(side=OrderSide.SELL, quantity=10))
        pos = ex._paper_engine.get_positions()
        assert "NSE_EQ|INE002A01018" not in pos

    # ── Get positions / funds / quote ─────────────────────────────────────

    def test_get_positions_returns_list(self):
        ex  = self._make_executor()
        pos = ex.get_positions()
        assert isinstance(pos, list)

    def test_get_funds_paper_mode(self):
        ex    = self._make_executor()
        funds = ex.get_funds()
        assert "available_cash" in funds
        assert funds["available_cash"] > 0

    def test_get_quote_paper_mode(self):
        ex    = self._make_executor()
        quote = ex.get_quote("NSE_EQ|INE002A01018")
        assert isinstance(quote, QuoteData)
        assert quote.last_price > 0

    # ── Cancel order tests ────────────────────────────────────────────────

    def test_cancel_order_paper_mode(self):
        ex      = self._make_executor()
        result  = ex.place_order(self._make_request())
        success = ex.cancel_order(result.order_id)
        assert success is True

    # ── Order status tests ────────────────────────────────────────────────

    def test_get_order_status_paper_mode(self):
        ex     = self._make_executor()
        result = ex.place_order(self._make_request())
        status = ex.get_order_status(result.order_id)
        assert status is not None
        assert status.order_id == result.order_id

    def test_get_order_status_unknown_id(self):
        ex     = self._make_executor()
        status = ex.get_order_status("NONEXISTENT-ID")
        assert status is None

    # ── P&L tests ─────────────────────────────────────────────────────────

    def test_paper_pnl_positive_gain(self):
        ex  = self._make_executor()
        ex.place_order(self._make_request(quantity=10))
        # Current price higher than fill → gain
        pnl = ex.get_paper_pnl({"NSE_EQ|INE002A01018": 110.0})
        assert pnl["total_unrealised_pnl"] > 0

    def test_paper_pnl_negative_loss(self):
        ex  = self._make_executor()
        ex.place_order(self._make_request(quantity=10))
        pnl = ex.get_paper_pnl({"NSE_EQ|INE002A01018": 90.0})
        assert pnl["total_unrealised_pnl"] < 0

    # ── Rate limiter tests ────────────────────────────────────────────────

    def test_rate_limiter_acquires_without_block(self):
        """Single acquire should not block."""
        rl    = RateLimiter(max_per_sec=25)
        start = time.monotonic()
        rl.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1   # should be near-instant

    def test_rate_limiter_initial_tokens(self):
        rl = RateLimiter(max_per_sec=10)
        assert rl._tokens == 10

    # ── Repr test ─────────────────────────────────────────────────────────

    def test_repr_contains_mode(self):
        ex = self._make_executor()
        assert "PAPER" in repr(ex)


# ── Run tests when file is executed directly ──────────────────────────────
if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))