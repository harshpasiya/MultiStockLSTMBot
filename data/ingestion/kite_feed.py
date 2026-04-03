"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Zerodha Kite Live Feed                 ║
║         Project : MultiStockLSTMBot                             ║
║         File    : data/ingestion/kite_feed.py                   ║
║         Phase   : 0 — Data Infrastructure                       ║
║         Purpose : Real-time WebSocket tick feed via Kite Connect ║
║                   Caches live prices in Redis                    ║
╚══════════════════════════════════════════════════════════════════╝

What this file does:
--------------------
1. Authenticates with Zerodha Kite Connect API
2. Opens a WebSocket connection to receive real-time tick data
3. Subscribes to all Nifty 500 instruments in FULL mode
   (FULL mode gives: OHLC, volume, bid/ask, OI for derivatives)
4. On every tick: writes latest price to Redis with 1-second TTL
5. Every minute: writes OHLCV candle to TimescaleDB (1-min bars)
6. Handles disconnects with automatic reconnection
7. Generates access token via browser-based OAuth flow

Kite Connect WebSocket Modes:
------------------------------
    LTP   : Last traded price only (lightest)
    QUOTE : LTP + OHLC + volume
    FULL  : QUOTE + market depth (bid/ask 5 levels) + OI  ← we use this

Redis Key Structure:
--------------------
    tick:{symbol}           → latest tick JSON (TTL: 5 seconds)
    ohlcv_1m:{symbol}:{ts} → 1-minute OHLCV accumulator

Usage:
------
    # Start live feed (run during market hours 9:15 AM - 3:30 PM IST)
    python -m data.ingestion.kite_feed --mode live

    # Generate new access token (run once per day before market open)
    python -m data.ingestion.kite_feed --mode auth

Dependencies:
-------------
    pip install kiteconnect redis python-dotenv loguru pyyaml
"""

import os
import json
import time
import argparse
import threading
import webbrowser

import redis
import yaml
import psycopg2
import psycopg2.extras

from datetime import datetime, date
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker

load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "kite_feed_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="7 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
)

# ── Constants ─────────────────────────────────────────────────────────────
MARKET_OPEN  = "09:15"
MARKET_CLOSE = "15:30"
TICK_TTL_SECONDS  = 5      # Redis TTL for live tick data
CANDLE_INTERVAL   = 60     # seconds — build 1-min candles
ACCESS_TOKEN_FILE = Path("config/.kite_access_token")  # cached daily token
INSTRUMENT_FILE   = Path("config/.kite_instruments.json")


# ══════════════════════════════════════════════════════════════════════════
#  AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════

def get_kite_client() -> KiteConnect:
    """
    Returns an authenticated KiteConnect client.
    Loads access token from cache file if available (tokens last 1 trading day).
    If token expired or missing, triggers browser-based re-authentication.
    """
    api_key    = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        raise EnvironmentError(
            "KITE_API_KEY and KITE_API_SECRET must be set in .env\n"
            "Get these from: https://developers.kite.trade/"
        )

    kite = KiteConnect(api_key=api_key)

    # Try loading cached access token
    if ACCESS_TOKEN_FILE.exists():
        token_data = json.loads(ACCESS_TOKEN_FILE.read_text())
        token_date = token_data.get("date")

        # Kite access tokens are valid for one trading day
        if token_date == str(date.today()):
            kite.set_access_token(token_data["access_token"])
            logger.info("Loaded cached Kite access token")
            return kite

    # Token expired or missing — generate new one
    logger.info("Access token expired or missing — starting OAuth flow...")
    return _generate_new_token(kite, api_key, api_secret)


def _generate_new_token(kite: KiteConnect, api_key: str, api_secret: str) -> KiteConnect:
    """
    Opens browser for Kite OAuth login.
    User logs in → Kite redirects to localhost with request_token.
    User pastes the request_token here → we exchange for access_token.
    """
    login_url = kite.login_url()
    logger.info(f"Opening browser for Kite login: {login_url}")
    webbrowser.open(login_url)

    print("\n" + "="*60)
    print("KITE AUTHENTICATION")
    print("="*60)
    print("1. Your browser has opened the Kite login page.")
    print("2. Log in with your Zerodha credentials.")
    print("3. After login, you will be redirected to a URL like:")
    print("   http://127.0.0.1/?request_token=XXXXXXXXXX&action=login&status=success")
    print("4. Copy the 'request_token' value from that URL.")
    print("="*60)

    request_token = input("Paste your request_token here: ").strip()

    try:
        session_data  = kite.generate_session(request_token, api_secret=api_secret)
        access_token  = session_data["access_token"]
        kite.set_access_token(access_token)

        # Cache the token for today
        ACCESS_TOKEN_FILE.parent.mkdir(exist_ok=True)
        ACCESS_TOKEN_FILE.write_text(json.dumps({
            "access_token": access_token,
            "date": str(date.today())
        }))

        logger.success("Kite authentication successful — access token cached")
        return kite

    except Exception as e:
        logger.error(f"Kite authentication failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════
#  INSTRUMENT LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_instrument_tokens(kite: KiteConnect, universe: set) -> dict:
    """
    Loads Kite instrument tokens for all symbols in the universe.
    Instrument tokens are numeric IDs used by the WebSocket — not symbols.

    Returns:
        Dict mapping instrument_token (int) → symbol (str)
        e.g. {738561: 'RELIANCE', 2953217: 'TCS', ...}

    Caches to config/.kite_instruments.json (refresh weekly or on new listings)
    """
    # Try cache first
    if INSTRUMENT_FILE.exists():
        cache = json.loads(INSTRUMENT_FILE.read_text())
        cache_date = cache.get("date", "")
        # Refresh weekly (instrument tokens rarely change)
        if cache_date == str(date.today()):
            token_map = {int(k): v for k, v in cache["tokens"].items()}
            logger.info(f"Loaded {len(token_map)} instrument tokens from cache")
            return token_map

    logger.info("Fetching instrument list from Kite...")

    # Download full NSE instrument list
    instruments = kite.instruments("NSE")

    token_map = {}
    for inst in instruments:
        symbol = inst["tradingsymbol"]
        if (
            symbol in universe
            and inst["segment"] == "NSE"
            and inst["instrument_type"] == "EQ"
        ):
            token_map[inst["instrument_token"]] = symbol

    # Cache
    INSTRUMENT_FILE.write_text(json.dumps({
        "date": str(date.today()),
        "tokens": {str(k): v for k, v in token_map.items()}
    }, indent=2))

    logger.info(f"Loaded {len(token_map)} instrument tokens for universe")
    return token_map


# ══════════════════════════════════════════════════════════════════════════
#  REDIS CLIENT
# ══════════════════════════════════════════════════════════════════════════

def get_redis_client() -> redis.Redis:
    """Returns authenticated Redis client from REDIS_URL in .env"""
    url = os.getenv("REDIS_URL", "redis://:godseye_redis_pass@localhost:6379")
    client = redis.from_url(url, decode_responses=True)
    client.ping()  # Verify connection
    return client


# ══════════════════════════════════════════════════════════════════════════
#  CANDLE BUILDER
# ══════════════════════════════════════════════════════════════════════════

class CandleBuilder:
    """
    Builds 1-minute OHLCV candles from raw tick data.
    Writes completed candles to TimescaleDB.

    How it works:
        - Accumulates ticks within the current minute
        - On minute boundary: finalizes candle and writes to DB
        - Thread-safe using locks
    """

    def __init__(self, db_conn):
        self.db_conn  = db_conn
        self.candles  = {}   # symbol → current open candle dict
        self.lock     = threading.Lock()

    def on_tick(self, symbol: str, price: float, volume: int, timestamp: datetime):
        """Called for every incoming tick — updates running candle"""
        minute_key = timestamp.strftime("%Y-%m-%d %H:%M:00")

        with self.lock:
            if symbol not in self.candles:
                # New candle
                self.candles[symbol] = {
                    "symbol"    : symbol,
                    "ts"        : minute_key,
                    "open"      : price,
                    "high"      : price,
                    "low"       : price,
                    "close"     : price,
                    "volume"    : volume,
                }
            else:
                candle = self.candles[symbol]

                if candle["ts"] != minute_key:
                    # Minute boundary crossed — flush old candle
                    self._flush_candle(candle)
                    # Start new candle
                    self.candles[symbol] = {
                        "symbol": symbol,
                        "ts"    : minute_key,
                        "open"  : price,
                        "high"  : price,
                        "low"   : price,
                        "close" : price,
                        "volume": volume,
                    }
                else:
                    # Update running candle
                    candle["high"]   = max(candle["high"], price)
                    candle["low"]    = min(candle["low"],  price)
                    candle["close"]  = price
                    candle["volume"] = volume  # Kite provides cumulative volume

    def _flush_candle(self, candle: dict):
        """Writes a completed 1-minute candle to TimescaleDB"""
        sql = """
            INSERT INTO intraday_ohlcv_1m
                (ts, symbol, open, high, low, close, volume)
            VALUES
                (%(ts)s, %(symbol)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
            ON CONFLICT (ts, symbol) DO UPDATE SET
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume;
        """
        try:
            with self.db_conn.cursor() as cur:
                cur.execute(sql, candle)
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to write candle for {candle['symbol']}: {e}")
            self.db_conn.rollback()

    def flush_all(self):
        """Flushes all open candles — call at market close"""
        with self.lock:
            for candle in self.candles.values():
                self._flush_candle(candle)
            self.candles.clear()
        logger.info("All open candles flushed to DB")


def ensure_intraday_table(conn):
    """Creates the 1-minute OHLCV hypertable if it doesn't exist"""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS intraday_ohlcv_1m (
                ts      TIMESTAMPTZ NOT NULL,
                symbol  VARCHAR(20) NOT NULL,
                open    NUMERIC(12,2),
                high    NUMERIC(12,2),
                low     NUMERIC(12,2),
                close   NUMERIC(12,2),
                volume  BIGINT,
                PRIMARY KEY (ts, symbol)
            );
        """)
        cur.execute("""
            SELECT create_hypertable(
                'intraday_ohlcv_1m', 'ts',
                if_not_exists => TRUE,
                migrate_data  => TRUE
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_intraday_symbol
            ON intraday_ohlcv_1m (symbol, ts DESC);
        """)
    conn.commit()
    logger.info("Intraday 1m table verified/created")


# ══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET FEED
# ══════════════════════════════════════════════════════════════════════════

class KiteLiveFeed:
    """
    Manages the Kite WebSocket connection.
    Handles reconnection, tick routing, and graceful shutdown.
    """

    def __init__(self, kite: KiteConnect, token_map: dict,
                 redis_client: redis.Redis, candle_builder: CandleBuilder):
        self.kite           = kite
        self.token_map      = token_map       # {instrument_token: symbol}
        self.tokens         = list(token_map.keys())
        self.redis          = redis_client
        self.candle_builder = candle_builder
        self.ticker         = None
        self.running        = False

    def start(self):
        """Starts the WebSocket connection"""
        api_key      = os.getenv("KITE_API_KEY")
        access_token = self.kite.access_token

        self.ticker = KiteTicker(api_key, access_token)

        # ── Attach event handlers ─────────────────────────────────────────
        self.ticker.on_ticks   = self._on_ticks
        self.ticker.on_connect = self._on_connect
        self.ticker.on_close   = self._on_close
        self.ticker.on_error   = self._on_error
        self.ticker.on_reconnect      = self._on_reconnect
        self.ticker.on_noreconnect    = self._on_noreconnect

        self.running = True
        logger.info(f"Starting WebSocket feed for {len(self.tokens)} instruments...")

        # reconnect=True → auto-reconnect on disconnect
        # interval=5     → retry every 5 seconds
        self.ticker.connect(threaded=True, reconnect=True, interval=5)

    def stop(self):
        """Gracefully stops the WebSocket connection"""
        self.running = False
        if self.ticker:
            self.ticker.close()
        self.candle_builder.flush_all()
        logger.info("WebSocket feed stopped")

    # ── WebSocket Event Handlers ──────────────────────────────────────────

    def _on_connect(self, ws, response):
        """Called when WebSocket connects — subscribe to all instruments"""
        logger.success(f"WebSocket connected — subscribing to {len(self.tokens)} instruments")
        # Subscribe in batches of 200 (Kite limit per subscription call)
        batch_size = 200
        for i in range(0, len(self.tokens), batch_size):
            batch = self.tokens[i:i + batch_size]
            self.ticker.subscribe(batch)
            self.ticker.set_mode(self.ticker.MODE_FULL, batch)
        logger.info("All instruments subscribed in FULL mode")

    def _on_ticks(self, ws, ticks: list):
        """
        Called on every tick (up to several times per second per instrument).
        Processes each tick: write to Redis + update candle builder.
        """
        now = datetime.now()

        for tick in ticks:
            token  = tick.get("instrument_token")
            symbol = self.token_map.get(token)

            if not symbol:
                continue

            price  = tick.get("last_price", 0)
            volume = tick.get("volume_traded", 0)

            if price <= 0:
                continue

            # ── Write to Redis (live price cache) ─────────────────────────
            tick_data = {
                "symbol"    : symbol,
                "price"     : price,
                "open"      : tick.get("ohlc", {}).get("open", price),
                "high"      : tick.get("ohlc", {}).get("high", price),
                "low"       : tick.get("ohlc", {}).get("low",  price),
                "volume"    : volume,
                "timestamp" : now.isoformat(),
            }
            self.redis.setex(
                f"tick:{symbol}",
                TICK_TTL_SECONDS,
                json.dumps(tick_data)
            )

            # ── Update candle builder ─────────────────────────────────────
            self.candle_builder.on_tick(symbol, price, volume, now)

    def _on_close(self, ws, code, reason):
        logger.warning(f"WebSocket closed: code={code} reason={reason}")

    def _on_error(self, ws, code, reason):
        logger.error(f"WebSocket error: code={code} reason={reason}")

    def _on_reconnect(self, ws, attempts_count):
        logger.info(f"WebSocket reconnecting... attempt {attempts_count}")

    def _on_noreconnect(self, ws):
        logger.critical("WebSocket max reconnection attempts reached — manual restart required")
        self.running = False


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def load_universe() -> set:
    """Loads Nifty 500 symbols from config/universe.yaml"""
    path = Path("config/universe.yaml")
    if not path.exists():
        raise FileNotFoundError(
            "config/universe.yaml not found.\n"
            "Run nse_bhavcopy.py first to generate it."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return set(data.get("nifty500", []))


def get_db_connection():
    """Returns psycopg2 connection to TimescaleDB"""
    url = os.getenv("TIMESCALE_URL")
    if not url:
        raise EnvironmentError("TIMESCALE_URL not set in .env")
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


def run_live():
    """
    Main entry point for live feed.
    Runs continuously during market hours.
    """
    logger.info("G.O.D.S E.Y.E — Kite Live Feed starting...")

    universe      = load_universe()
    kite          = get_kite_client()
    token_map     = load_instrument_tokens(kite, universe)
    redis_client  = get_redis_client()
    db_conn       = get_db_connection()

    ensure_intraday_table(db_conn)

    candle_builder = CandleBuilder(db_conn)
    feed           = KiteLiveFeed(kite, token_map, redis_client, candle_builder)

    try:
        feed.start()
        logger.info("Feed running — press Ctrl+C to stop")

        # Keep main thread alive
        while feed.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        feed.stop()
        db_conn.close()
        logger.info("Kite feed shut down cleanly")


def run_auth():
    """Generates and caches a new Kite access token"""
    logger.info("Starting Kite authentication flow...")
    get_kite_client()
    logger.success("Authentication complete — token cached for today")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — Kite Live Feed")
    parser.add_argument(
        "--mode",
        choices=["live", "auth"],
        required=True,
        help="live: start WebSocket feed | auth: generate new access token"
    )
    args = parser.parse_args()

    if args.mode == "live":
        run_live()
    elif args.mode == "auth":
        run_auth()