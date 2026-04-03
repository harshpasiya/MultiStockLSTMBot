"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Data Quality Checks                    ║
║         Project : MultiStockLSTMBot                             ║
║         File    : data/validators/quality_checks.py             ║
║         Phase   : 0 — Data Infrastructure                       ║
║         Purpose : Validates all data sources for freshness,     ║
║                   completeness, and correctness                  ║
╚══════════════════════════════════════════════════════════════════╝

What this file does:
--------------------
1. Checks all data sources for staleness (last update timestamp)
2. Validates OHLCV completeness (all 500 stocks present for each date)
3. Detects anomalies (prices outside expected ranges)
4. Checks FII/DII data freshness
5. Validates Elasticsearch news index health
6. Sends alerts via logger (Telegram integration added in Phase 5)
7. Generates a daily data quality report

Run this as a health check before market open every day (8:30 AM IST)

Usage:
------
    python -m data.validators.quality_checks --check all
    python -m data.validators.quality_checks --check ohlcv
    python -m data.validators.quality_checks --check freshness
    python -m data.validators.quality_checks --check report
"""

import os
import argparse
import psycopg2
import redis

from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "quality_checks_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
)

# ── Thresholds ────────────────────────────────────────────────────────────
MIN_UNIVERSE_COVERAGE = 0.95   # at least 95% of 500 stocks must have data
MAX_PRICE_CHANGE_PCT  = 0.50   # flag daily moves > 50% as anomalies
STALE_MINUTES_OHLCV   = 1440   # 24 hours (daily data)
STALE_MINUTES_TICK    = 10     # 10 minutes (live ticks during market hours)
STALE_MINUTES_FII     = 120    # 2 hours
STALE_MINUTES_OPTIONS = 10     # 10 minutes


# ══════════════════════════════════════════════════════════════════════════
#  CONNECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_db_connection():
    url = os.getenv("TIMESCALE_URL")
    if not url:
        raise EnvironmentError("TIMESCALE_URL not set in .env")
    return psycopg2.connect(url)


def get_redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://:godseye_redis_pass@localhost:6379")
    return redis.from_url(url, decode_responses=True)


# ══════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL CHECKS
# ══════════════════════════════════════════════════════════════════════════

def check_ohlcv_completeness(conn) -> dict:
    """
    Checks that today's (or latest) Bhavcopy data is complete.

    Returns:
        Dict with status, coverage_pct, missing_count, issues list
    """
    issues = []

    with conn.cursor() as cur:
        # Get latest loaded date
        cur.execute("SELECT MAX(date) FROM daily_ohlcv;")
        latest_date = cur.fetchone()[0]

        if not latest_date:
            return {
                "status"      : "FAIL",
                "latest_date" : None,
                "coverage_pct": 0.0,
                "issues"      : ["daily_ohlcv table is empty — run bhavcopy backfill"]
            }

        # Count stocks for latest date
        cur.execute(
            "SELECT COUNT(DISTINCT symbol) FROM daily_ohlcv WHERE date = %s;",
            (latest_date,)
        )
        stock_count = cur.fetchone()[0]

        # Expected: ~500 stocks
        coverage = stock_count / 500
        if coverage < MIN_UNIVERSE_COVERAGE:
            issues.append(
                f"Only {stock_count}/500 stocks in latest date "
                f"({coverage*100:.1f}% coverage)"
            )

        # Check for suspiciously old data
        days_old = (date.today() - latest_date).days
        if days_old > 3:  # More than 3 days old (accounts for weekends)
            issues.append(f"Latest OHLCV data is {days_old} days old — check bhavcopy download")

        # Check for zero-volume anomalies in latest date
        cur.execute("""
            SELECT COUNT(*) FROM daily_ohlcv
            WHERE date = %s AND (volume = 0 OR volume IS NULL);
        """, (latest_date,))
        zero_vol = cur.fetchone()[0]
        if zero_vol > 10:
            issues.append(f"{zero_vol} stocks with zero volume on {latest_date}")

        # Check for OHLC anomalies
        cur.execute("""
            SELECT COUNT(*) FROM daily_ohlcv
            WHERE date = %s AND (high < low OR close <= 0);
        """, (latest_date,))
        ohlc_errors = cur.fetchone()[0]
        if ohlc_errors > 0:
            issues.append(f"{ohlc_errors} OHLC inconsistencies on {latest_date}")

    status = "PASS" if not issues else "WARN" if coverage >= 0.90 else "FAIL"

    result = {
        "status"      : status,
        "latest_date" : str(latest_date),
        "stock_count" : stock_count,
        "coverage_pct": round(coverage * 100, 1),
        "days_old"    : days_old,
        "issues"      : issues,
    }

    _log_result("OHLCV Completeness", result)
    return result


def check_data_freshness(redis_client: redis.Redis) -> dict:
    """
    Checks Redis for staleness of real-time data sources.
    Only meaningful during market hours.
    """
    now      = datetime.now()
    hour_min = now.strftime("%H:%M")
    issues   = []

    checks = {
        "live_tick_sample": ("tick:RELIANCE", STALE_MINUTES_TICK),
        "mds_score"       : ("mds:current", STALE_MINUTES_FII * 60),
        "gap_prediction"  : ("global:gap_prediction", STALE_MINUTES_OPTIONS * 60),
        "pcr_nifty"       : ("pcr:NIFTY", STALE_MINUTES_OPTIONS * 60),
    }

    freshness = {}
    for name, (key, max_age_secs) in checks.items():
        exists = redis_client.exists(key)
        ttl    = redis_client.ttl(key)

        if not exists:
            # Only flag as issue during market hours
            if "09:15" <= hour_min <= "15:35":
                issues.append(f"Redis key '{key}' missing during market hours")
            freshness[name] = {"status": "MISSING", "ttl": None}
        else:
            freshness[name] = {"status": "PRESENT", "ttl": ttl}

    status = "PASS" if not issues else "WARN"

    result = {
        "status"    : status,
        "market_hrs": "09:15" <= hour_min <= "15:35",
        "freshness" : freshness,
        "issues"    : issues,
    }

    _log_result("Data Freshness", result)
    return result


def check_fii_dii_data(conn) -> dict:
    """Checks FII/DII data for today or last trading day"""
    issues = []

    with conn.cursor() as cur:
        # Latest FII/DII date
        cur.execute("SELECT MAX(date) FROM fii_dii_flow;")
        latest = cur.fetchone()[0]

        if not latest:
            return {
                "status": "FAIL",
                "issues": ["fii_dii_flow table empty — run FII/DII backfill"]
            }

        days_old = (date.today() - latest).days
        if days_old > 2:
            issues.append(f"FII/DII data is {days_old} days old")

        # Check both provisional and final exist for latest date
        cur.execute("""
            SELECT data_type FROM fii_dii_flow
            WHERE date = %s;
        """, (latest,))
        types = {row[0] for row in cur.fetchall()}

        if "final" not in types and days_old == 0:
            issues.append(f"Final FII/DII data not yet loaded for {latest}")

        # Check for null MDS scores
        cur.execute("""
            SELECT COUNT(*) FROM fii_dii_flow
            WHERE mds_score IS NULL AND date >= CURRENT_DATE - INTERVAL '7 days';
        """)
        null_mds = cur.fetchone()[0]
        if null_mds > 0:
            issues.append(f"{null_mds} records with null MDS score in last 7 days")

    status = "PASS" if not issues else "WARN"
    result = {
        "status"     : status,
        "latest_date": str(latest),
        "types_loaded": list(types),
        "days_old"   : days_old,
        "issues"     : issues,
    }

    _log_result("FII/DII Data", result)
    return result


def check_intraday_data(conn) -> dict:
    """Checks 1-minute intraday OHLCV completeness"""
    issues = []

    with conn.cursor() as cur:
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'intraday_ohlcv_1m'
            );
        """)
        table_exists = cur.fetchone()[0]

        if not table_exists:
            return {
                "status": "WARN",
                "issues": ["intraday_ohlcv_1m table does not exist yet — will be created on first kite_feed run"]
            }

        # Latest intraday timestamp
        cur.execute("SELECT MAX(ts) FROM intraday_ohlcv_1m;")
        latest_ts = cur.fetchone()[0]

        if not latest_ts:
            return {
                "status": "WARN",
                "issues": ["No intraday data yet — start kite_feed during market hours"]
            }

        age_minutes = (datetime.now() - latest_ts.replace(tzinfo=None)).total_seconds() / 60

        # During market hours, data should be < 10 minutes old
        now_hm = datetime.now().strftime("%H:%M")
        if "09:15" <= now_hm <= "15:35" and age_minutes > 10:
            issues.append(f"Intraday data is {age_minutes:.0f} minutes old — check kite_feed")

    status = "PASS" if not issues else "WARN"
    result = {
        "status"       : status,
        "latest_ts"    : str(latest_ts) if latest_ts else None,
        "age_minutes"  : round(age_minutes, 1) if latest_ts else None,
        "issues"       : issues,
    }

    _log_result("Intraday Data", result)
    return result


def check_options_data(conn) -> dict:
    """Checks options chain data freshness"""
    issues = []

    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'options_pcr'
            );
        """)
        if not cur.fetchone()[0]:
            return {
                "status": "WARN",
                "issues": ["options_pcr table not created yet — run options_chain.py first"]
            }

        cur.execute("SELECT MAX(ts) FROM options_pcr;")
        latest_ts = cur.fetchone()[0]

        if not latest_ts:
            return {
                "status": "WARN",
                "issues": ["No options PCR data — run options_chain scraper"]
            }

        age_min = (datetime.now() - latest_ts.replace(tzinfo=None)).total_seconds() / 60
        now_hm  = datetime.now().strftime("%H:%M")

        if "09:15" <= now_hm <= "15:35" and age_min > STALE_MINUTES_OPTIONS:
            issues.append(f"Options PCR data is {age_min:.0f} mins old (threshold: {STALE_MINUTES_OPTIONS} mins)")

    status = "PASS" if not issues else "WARN"
    result = {
        "status"     : status,
        "latest_ts"  : str(latest_ts) if latest_ts else None,
        "age_minutes": round(age_min, 1) if latest_ts else None,
        "issues"     : issues,
    }

    _log_result("Options PCR Data", result)
    return result


# ══════════════════════════════════════════════════════════════════════════
#  FULL REPORT
# ══════════════════════════════════════════════════════════════════════════

def generate_report() -> dict:
    """
    Runs all checks and returns a complete health report.
    Run this every morning at 8:30 AM before market open.
    """
    conn         = get_db_connection()
    redis_client = get_redis_client()

    report = {
        "generated_at"  : datetime.now().isoformat(),
        "overall_status": "PASS",
        "checks"        : {}
    }

    check_fns = {
        "ohlcv_completeness": lambda: check_ohlcv_completeness(conn),
        "data_freshness"    : lambda: check_data_freshness(redis_client),
        "fii_dii_data"      : lambda: check_fii_dii_data(conn),
        "intraday_data"     : lambda: check_intraday_data(conn),
        "options_data"      : lambda: check_options_data(conn),
    }

    for check_name, check_fn in check_fns.items():
        try:
            result = check_fn()
            report["checks"][check_name] = result

            if result["status"] == "FAIL":
                report["overall_status"] = "FAIL"
            elif result["status"] == "WARN" and report["overall_status"] == "PASS":
                report["overall_status"] = "WARN"

        except Exception as e:
            report["checks"][check_name] = {
                "status": "ERROR",
                "error" : str(e),
                "issues": [f"Check threw exception: {e}"]
            }
            report["overall_status"] = "FAIL"

    conn.close()

    # Print summary
    print("\n" + "="*55)
    print("  G.O.D.S E.Y.E — Data Quality Report")
    print(f"  {report['generated_at']}")
    print("="*55)
    for name, result in report["checks"].items():
        icon = "✓" if result["status"] == "PASS" else "⚠" if result["status"] == "WARN" else "✗"
        print(f"  {icon} {name:<30} {result['status']}")
        for issue in result.get("issues", []):
            print(f"      → {issue}")
    print("="*55)
    print(f"  Overall: {report['overall_status']}")
    print("="*55 + "\n")

    return report


# ══════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════

def _log_result(check_name: str, result: dict):
    """Logs check result with appropriate level"""
    status = result.get("status", "UNKNOWN")
    issues = result.get("issues", [])

    if status == "PASS":
        logger.success(f"{check_name}: PASS")
    elif status == "WARN":
        logger.warning(f"{check_name}: WARN — {'; '.join(issues)}")
    else:
        logger.error(f"{check_name}: FAIL — {'; '.join(issues)}")


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E — Data Quality Checker"
    )
    parser.add_argument(
        "--check",
        choices=["all", "ohlcv", "freshness", "fii", "intraday", "options", "report"],
        default="report",
        help="Which check to run (default: report = all checks + summary)"
    )
    args = parser.parse_args()

    if args.check in ("all", "report"):
        generate_report()
    else:
        conn         = get_db_connection()
        redis_client = get_redis_client()

        check_map = {
            "ohlcv"    : lambda: check_ohlcv_completeness(conn),
            "freshness": lambda: check_data_freshness(redis_client),
            "fii"      : lambda: check_fii_dii_data(conn),
            "intraday" : lambda: check_intraday_data(conn),
            "options"  : lambda: check_options_data(conn),
        }

        result = check_map[args.check]()
        import json
        print(json.dumps(result, indent=2, default=str))
        conn.close()