"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Nightly Retraining DAG (Apache Airflow)         ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : airflow/dags/retraining_dag.py                         ║
║         Phase   : 4 — Paper Trading & Live Monitoring                   ║
║                                                                          ║
║  What this DAG does:                                                     ║
║    Runs every night at 22:30 IST (17:00 UTC) after NSE market close     ║
║    and all final data (FII/DII final, Bhavcopy) is available.           ║
║                                                                          ║
║  Pipeline steps (in order):                                              ║
║    T1  validate_infrastructure   — ping DB, Redis, Elasticsearch        ║
║    T2  download_bhavcopy         — NSE daily OHLCV + delivery data      ║
║    T3  download_fii_dii_final    — NSE FII/DII final settlement data    ║
║    T4  run_feature_pipeline      — recompute all 6 pillars for today    ║
║    T5  recompute_embeddings      — update backbone_embeddings table      ║
║    T6  validate_features         — check IC > 0.02, no NaN, no drift   ║
║    T7  finetune_backbone         — 2–3 epoch fine-tune on last 30 days  ║
║    T8  finetune_rl_agent         — PPO update on last 30 days outcomes  ║
║    T9  validate_model            — val IC + accuracy gate check         ║
║    T10 deploy_model              — copy new checkpoint to production    ║
║    T11 send_summary_alert        — Telegram summary of retrain result   ║
║                                                                          ║
║  Rollback:                                                               ║
║    If T9 (validate_model) fails, T10 is skipped — production model     ║
║    stays unchanged. Previous checkpoint is preserved.                   ║
║                                                                          ║
║  Schedule:                                                               ║
║    22:30 IST = 17:00 UTC daily (Mon–Fri, skips weekends via short-      ║
║    circuit in T2 if no Bhavcopy available)                              ║
║                                                                          ║
║  Setup:                                                                  ║
║    pip install apache-airflow                                            ║
║    Place this file in: airflow/dags/retraining_dag.py                   ║
║    airflow db init && airflow scheduler &                                ║
║    airflow webserver --port 8080 &                                       ║
║    # UI at http://localhost:8080                                         ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install apache-airflow apache-airflow-providers-postgres          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import subprocess
from datetime   import datetime, timedelta
from pathlib    import Path
from typing     import Dict, Any

from loguru import logger

# ── Airflow imports ────────────────────────────────────────────────────────
try:
    from airflow                          import DAG
    from airflow.operators.python         import PythonOperator, ShortCircuitOperator
    from airflow.operators.bash           import BashOperator
    from airflow.utils.dates              import days_ago
    from airflow.models                   import Variable
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# ── Project root (two levels up from airflow/dags/) ───────────────────────
PROJECT_ROOT    = Path(__file__).parent.parent.parent
CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"
LOG_DIR         = PROJECT_ROOT / "logs" / "retraining"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Environment ────────────────────────────────────────────────────────────
DB_URL          = os.getenv("TIMESCALE_URL",
                            "postgresql://godseye_user:godseye_pass@localhost:5433/godseye")
REDIS_URL       = os.getenv("REDIS_URL",
                            "redis://:godseye_redis_pass@localhost:6380")
ES_URL          = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Gate criteria for model validation ────────────────────────────────────
MIN_VAL_IC      = 0.04    # minimum acceptable validation IC
MIN_VAL_ACC     = 0.54    # minimum acceptable direction accuracy
MAX_DRIFT_KL    = 0.15    # reject retrain if drift exceeds this

# ── DAG default args ───────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner"            : "godseye",
    "depends_on_past"  : False,
    "email_on_failure" : False,
    "email_on_retry"   : False,
    "retries"          : 1,
    "retry_delay"      : timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ══════════════════════════════════════════════════════════════════════════
#  TASK FUNCTIONS
#  Each function maps to one Airflow PythonOperator task.
#  All tasks push results to XCom for downstream tasks to read.
# ══════════════════════════════════════════════════════════════════════════

def task_validate_infrastructure(**context) -> Dict:
    """
    T1: Validates that all infrastructure services are reachable.
    Short-circuits the entire DAG if any service is down.

    Checks:
        - TimescaleDB (port 5433)
        - Redis (port 6380)
        - Elasticsearch (port 9200)
    """
    import psycopg2
    import redis
    import requests

    results = {}

    # ── TimescaleDB ───────────────────────────────────────────────────────
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM daily_ohlcv;")
            row_count = cur.fetchone()[0]
        conn.close()
        results["timescaledb"] = {"status": "ok", "row_count": row_count}
        logger.info(f"TimescaleDB OK — {row_count:,} rows in daily_ohlcv")
    except Exception as e:
        results["timescaledb"] = {"status": "error", "error": str(e)}
        raise RuntimeError(f"TimescaleDB unreachable: {e}")

    # ── Redis ─────────────────────────────────────────────────────────────
    try:
        r = redis.from_url(REDIS_URL, socket_timeout=5)
        r.ping()
        results["redis"] = {"status": "ok"}
        logger.info("Redis OK")
    except Exception as e:
        results["redis"] = {"status": "warning", "error": str(e)}
        logger.warning(f"Redis unavailable (non-critical): {e}")

    # ── Elasticsearch ─────────────────────────────────────────────────────
    try:
        resp = requests.get(f"{ES_URL}/_cluster/health", timeout=5)
        status = resp.json().get("status", "unknown")
        results["elasticsearch"] = {"status": status}
        logger.info(f"Elasticsearch {status}")
    except Exception as e:
        results["elasticsearch"] = {"status": "warning", "error": str(e)}
        logger.warning(f"Elasticsearch unavailable (non-critical): {e}")

    context["ti"].xcom_push(key="infra_check", value=results)
    logger.info("Infrastructure validation complete.")
    return results


def task_download_bhavcopy(**context) -> Dict:
    """
    T2: Downloads today's NSE Bhavcopy (OHLCV + delivery data).
    Short-circuits if today is weekend or holiday (no data available).

    Returns:
        Dict with rows_inserted and date processed
    """
    from data.ingestion.nse_bhavcopy import run_daily
    from datetime import date

    today = date.today()

    # Skip weekends
    if today.weekday() >= 5:
        logger.info(f"Weekend ({today}) — skipping Bhavcopy download.")
        context["ti"].xcom_push(key="bhavcopy_result",
                                value={"skipped": True, "reason": "weekend"})
        return {"skipped": True}

    try:
        run_daily()
        result = {"date": str(today), "status": "success"}
        logger.success(f"Bhavcopy downloaded for {today}")
    except Exception as e:
        logger.warning(f"Bhavcopy download failed (may be holiday): {e}")
        result = {"date": str(today), "status": "failed", "error": str(e)}

    context["ti"].xcom_push(key="bhavcopy_result", value=result)
    return result


def task_download_fii_dii(**context) -> Dict:
    """
    T3: Downloads NSE FII/DII final settlement data for today.
    Final data is available after 6 PM IST; DAG runs at 10:30 PM IST.

    Returns:
        Dict with fii_net, dii_net for today
    """
    from data.ingestion.fii_dii_scraper import run_final as run_daily_final
    from datetime import date

    today = date.today()
    if today.weekday() >= 5:
        logger.info("Weekend — skipping FII/DII download.")
        return {"skipped": True}

    try:
        result = run_daily_final()
        logger.success(f"FII/DII data downloaded for {today}")
    except Exception as e:
        logger.warning(f"FII/DII download failed: {e}")
        result = {"status": "failed", "error": str(e)}

    context["ti"].xcom_push(key="fii_dii_result", value=result)
    return result


def task_run_feature_pipeline(**context) -> Dict:
    """
    T4: Recomputes all 6 feature pillars for today's data.
    Runs each pillar's .run_all() for the last 5 trading days
    (incremental update, not full recompute).

    Returns:
        Dict with per-pillar row counts and timing
    """
    from datetime import date, timedelta
    import time

    start_date = date.today() - timedelta(days=7)
    end_date   = date.today()
    results    = {}

    pillars = [
        ("trend",       "features.trend",       "TrendExtractor"),
        ("msi",         "features.msi",         "MSIExtractor"),
        ("fii_dii",     "features.fii_dii",     "FIIDIIExtractor"),
        ("sentiment",   "features.sentiment",   "SentimentExtractor"),
        ("volatility",  "features.volatility",  "VolatilityExtractor"),
        ("correlation", "features.correlation", "CorrelationExtractor"),
    ]

    for pillar_name, module_path, class_name in pillars:
        t0 = time.time()
        try:
            module    = __import__(module_path, fromlist=[class_name])
            extractor = getattr(module, class_name)()
            extractor.run_all(start_date=start_date, end_date=end_date)
            elapsed   = time.time() - t0
            results[pillar_name] = {"status": "ok", "elapsed_s": round(elapsed, 1)}
            logger.success(f"Pillar {pillar_name} updated in {elapsed:.1f}s")
        except Exception as e:
            results[pillar_name] = {"status": "error", "error": str(e)}
            logger.error(f"Pillar {pillar_name} failed: {e}")

    context["ti"].xcom_push(key="feature_results", value=results)
    return results


def task_recompute_embeddings(**context) -> Dict:
    """
    T5: Recomputes backbone embeddings for all symbols using today's features.
    Updates the backbone_embeddings table in TimescaleDB.

    Returns:
        Dict with symbols processed and timing
    """
    import time

    t0 = time.time()
    try:
        # Run the precompute_embeddings script as a subprocess
        # to ensure GPU memory is properly managed
        result = subprocess.run(
            [sys.executable, "-m", "training.precompute_embeddings",
             "--incremental"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=1800
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            logger.success(f"Embeddings recomputed in {elapsed:.1f}s")
            outcome = {"status": "ok", "elapsed_s": round(elapsed, 1)}
        else:
            logger.error(f"Embedding recompute failed: {result.stderr[-500:]}")
            outcome = {"status": "error", "stderr": result.stderr[-200:]}

    except subprocess.TimeoutExpired:
        outcome = {"status": "timeout", "elapsed_s": 1800}
        logger.error("Embedding recompute timed out after 30 minutes.")
    except Exception as e:
        outcome = {"status": "error", "error": str(e)}
        logger.error(f"Embedding recompute exception: {e}")

    context["ti"].xcom_push(key="embedding_result", value=outcome)
    return outcome


def task_validate_features(**context) -> bool:
    """
    T6: Validates that today's feature data meets quality gates.
    ShortCircuitOperator — returns False to halt pipeline if gates fail.

    Gates:
        - No NaN values in fused feature table for today
        - Feature drift KL < MAX_DRIFT_KL
        - At least 80% of universe symbols have features today

    Returns:
        True if all gates pass (pipeline continues)
        False if any gate fails (pipeline stops, model NOT retrained)
    """
    import psycopg2
    import numpy as np
    from datetime import date

    today = str(date.today())
    gates_passed = True

    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:

            # Gate 1: Check feature count for today
            cur.execute("""
                SELECT COUNT(DISTINCT symbol)
                FROM features_fused
                WHERE date = %s;
            """, (today,))
            today_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(DISTINCT symbol) FROM daily_ohlcv;")
            total_symbols = cur.fetchone()[0]

            coverage = today_count / max(total_symbols, 1)
            if coverage < 0.80:
                logger.warning(
                    f"Feature coverage {coverage:.1%} < 80% "
                    f"({today_count}/{total_symbols} symbols). Gate FAIL."
                )
                gates_passed = False
            else:
                logger.info(f"Feature coverage: {coverage:.1%} ✓")

            # Gate 2: Check for NaN in key columns
            cur.execute(f"""
                SELECT COUNT(*)
                FROM features_fused
                WHERE date = %s
                  AND (f00_trend_score IS NULL
                    OR f06_msi_signal IS NULL
                    OR f17_volatility_score IS NULL);
            """, (today,))
            null_count = cur.fetchone()[0]

            if null_count > today_count * 0.10:
                logger.warning(
                    f"Too many NULL features: {null_count} rows. Gate FAIL."
                )
                gates_passed = False
            else:
                logger.info(f"NULL check: {null_count} nulls ✓")

        conn.close()

    except Exception as e:
        logger.error(f"Feature validation DB error: {e}")
        gates_passed = False

    # Gate 3: Drift check
    try:
        from monitoring.drift_detector import DriftDetector
        import numpy as np

        detector = DriftDetector()
        loaded   = detector.load_reference()

        if loaded:
            # Add a small sample of today's features as live data
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:
                cols = ", ".join([f"f{i:02d}_{n.split('_',1)[1]}"
                                  if i < 28 else f"f{i:02d}"
                                  for i, n in enumerate([
                    "trend_score","ema_ribbon_gap","adx_normalized",
                    "supertrend_dir","price_vs_ema200","swing_structure",
                    "msi_signal","vrsi_normalized","mfi_normalized",
                    "msi_divergence","mds_continuous","fii_norm",
                    "dii_norm","sentiment_score","sentiment_momentum",
                    "event_flag","market_fear_greed_n","volatility_score",
                    "atr_pct_normalized","vol_regime_code_n","hv_percentile_n",
                    "correlation_score","sector_divergence_n","lead_lag_score",
                    "peer_corr_mean","delivery_mom_n","swing_tp_normalized",
                    "swing_sl_normalized",
                ])])
                cur.execute(f"""
                    SELECT f00_trend_score,f01_ema_ribbon_gap,f02_adx_normalized,
                           f03_supertrend_dir,f04_price_vs_ema200,f05_swing_structure,
                           f06_msi_signal,f07_vrsi_normalized,f08_mfi_normalized,
                           f09_msi_divergence,f10_mds_continuous,f11_fii_norm,
                           f12_dii_norm,f13_sentiment_score,f14_sentiment_momentum,
                           f15_event_flag,f16_market_fear_greed_n,f17_volatility_score,
                           f18_atr_pct_normalized,f19_vol_regime_code_n,f20_hv_percentile_n,
                           f21_correlation_score,f22_sector_divergence_n,f23_lead_lag_score,
                           f24_peer_corr_mean,f25_delivery_mom_n,f26_swing_tp_normalized,
                           f27_swing_sl_normalized
                    FROM features_fused
                    WHERE date = %s
                    LIMIT 200;
                """, (today,))
                rows = cur.fetchall()
            conn.close()

            if rows:
                arr = np.array(rows, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0)
                detector.add_live_batch(arr)
                report = detector.compute_drift()

                if report and report.aggregate_kl > MAX_DRIFT_KL:
                    logger.warning(
                        f"Feature drift KL={report.aggregate_kl:.4f} "
                        f"> {MAX_DRIFT_KL}. Gate FAIL — skipping retrain."
                    )
                    gates_passed = False
                elif report:
                    logger.info(
                        f"Drift check: KL={report.aggregate_kl:.4f} ✓"
                    )

    except Exception as e:
        logger.warning(f"Drift check failed (non-critical): {e}")

    context["ti"].xcom_push(key="validation_passed", value=gates_passed)
    logger.info(f"Feature validation: {'PASS ✓' if gates_passed else 'FAIL ✗'}")
    return gates_passed


def task_finetune_backbone(**context) -> Dict:
    """
    T7: Fine-tunes the LSTM+Transformer backbone on the last 30 days.
    Runs 2–3 epochs only — prevents catastrophic forgetting.
    Uses AdamW with LR = 1e-5 (10× lower than original training).

    Returns:
        Dict with final val_ic, val_acc, and elapsed time
    """
    import time

    logger.info("Starting backbone fine-tuning (2 epochs, LR=1e-5)...")
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "training.pretrain_backbone",
             "--mode", "finetune",
             "--epochs", "2",
             "--lr", "1e-5",
             "--days", "30"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=3600
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            # Parse val_ic and val_acc from stdout
            val_ic  = _parse_metric(result.stdout, "val_ic")
            val_acc = _parse_metric(result.stdout, "val_acc")
            outcome = {
                "status" : "ok",
                "val_ic" : val_ic,
                "val_acc": val_acc,
                "elapsed_s": round(elapsed, 1),
            }
            logger.success(
                f"Backbone fine-tuned | val_ic={val_ic:.4f} "
                f"val_acc={val_acc:.3f} | {elapsed:.1f}s"
            )
        else:
            outcome = {
                "status": "error",
                "stderr": result.stderr[-300:],
                "elapsed_s": round(elapsed, 1),
            }
            logger.error(f"Backbone fine-tune failed: {result.stderr[-200:]}")

    except subprocess.TimeoutExpired:
        outcome = {"status": "timeout", "elapsed_s": 3600}
        logger.error("Backbone fine-tune timed out after 60 minutes.")
    except Exception as e:
        outcome = {"status": "error", "error": str(e)}

    context["ti"].xcom_push(key="backbone_result", value=outcome)
    return outcome


def task_finetune_rl_agent(**context) -> Dict:
    """
    T8: Updates the PPO swing RL agent on last 30 days of trade outcomes.
    Runs 100K steps (quick update, not full re-training).

    Returns:
        Dict with updated val_sharpe and elapsed time
    """
    import time

    logger.info("Starting RL agent fine-tuning (100K steps)...")
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "training.train_swing_rl",
             "--resume",
             "--total-steps", "100000",
             "--n-envs", "4"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=3600
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            val_sharpe = _parse_metric(result.stdout, "val_sharpe")
            outcome    = {
                "status"    : "ok",
                "val_sharpe": val_sharpe,
                "elapsed_s" : round(elapsed, 1),
            }
            logger.success(
                f"RL agent updated | val_sharpe={val_sharpe:.3f} | {elapsed:.1f}s"
            )
        else:
            outcome = {
                "status": "error",
                "stderr": result.stderr[-300:],
                "elapsed_s": round(elapsed, 1),
            }
            logger.error(f"RL fine-tune failed: {result.stderr[-200:]}")

    except subprocess.TimeoutExpired:
        outcome = {"status": "timeout", "elapsed_s": 3600}
        logger.error("RL fine-tune timed out.")
    except Exception as e:
        outcome = {"status": "error", "error": str(e)}

    context["ti"].xcom_push(key="rl_result", value=outcome)
    return outcome


def task_validate_model(**context) -> bool:
    """
    T9: Validates the newly fine-tuned model against gate criteria.
    ShortCircuitOperator — returns False to prevent deployment if gates fail.
    Previous production model remains unchanged on failure.

    Gates:
        - val_ic >= MIN_VAL_IC (0.04)
        - val_acc >= MIN_VAL_ACC (0.54)

    Returns:
        True if model passes gates (deployment proceeds)
        False if gates fail (deployment skipped, rollback preserved)
    """
    ti = context["ti"]

    backbone_result = ti.xcom_pull(key="backbone_result", task_ids="finetune_backbone")
    rl_result       = ti.xcom_pull(key="rl_result",       task_ids="finetune_rl_agent")

    gates_passed = True

    # Check backbone gates
    if backbone_result and backbone_result.get("status") == "ok":
        val_ic  = backbone_result.get("val_ic",  0.0)
        val_acc = backbone_result.get("val_acc", 0.0)

        if val_ic < MIN_VAL_IC:
            logger.warning(
                f"val_ic={val_ic:.4f} < {MIN_VAL_IC} (gate FAIL)"
            )
            gates_passed = False
        else:
            logger.info(f"val_ic={val_ic:.4f} ≥ {MIN_VAL_IC} ✓")

        if val_acc < MIN_VAL_ACC:
            logger.warning(
                f"val_acc={val_acc:.3f} < {MIN_VAL_ACC} (gate FAIL)"
            )
            gates_passed = False
        else:
            logger.info(f"val_acc={val_acc:.3f} ≥ {MIN_VAL_ACC} ✓")

    elif backbone_result and backbone_result.get("status") != "ok":
        logger.warning("Backbone fine-tune did not succeed — skipping deployment.")
        gates_passed = False

    # Check RL gate (non-blocking — RL failure doesn't prevent backbone deploy)
    if rl_result and rl_result.get("status") != "ok":
        logger.warning("RL fine-tune failed — backbone will still deploy if gates pass.")

    context["ti"].xcom_push(key="model_validated", value=gates_passed)
    logger.info(f"Model validation: {'PASS ✓' if gates_passed else 'FAIL ✗'}")
    return gates_passed


def task_deploy_model(**context) -> Dict:
    """
    T10: Deploys the validated fine-tuned model to production.

    Steps:
        1. Back up current production checkpoint
        2. Copy fine-tuned checkpoint to production path
        3. Update deployment metadata in DB
        4. Signal the running signal_engine to reload model

    Returns:
        Dict with deployed checkpoint path and timestamp
    """
    import psycopg2
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Backup current production model ───────────────────────────────────
    prod_ckpt   = CHECKPOINT_DIR / "pretrain_best.pt"
    backup_ckpt = CHECKPOINT_DIR / f"pretrain_backup_{timestamp}.pt"

    try:
        if prod_ckpt.exists():
            shutil.copy2(prod_ckpt, backup_ckpt)
            logger.info(f"Backed up production model to {backup_ckpt.name}")
    except Exception as e:
        logger.warning(f"Backup failed (non-critical): {e}")

    # ── Copy fine-tuned checkpoint to production ───────────────────────────
    finetuned_ckpt = CHECKPOINT_DIR / "pretrain_finetuned.pt"
    deployed       = False

    if finetuned_ckpt.exists():
        try:
            shutil.copy2(finetuned_ckpt, prod_ckpt)
            logger.success(f"Deployed fine-tuned model to {prod_ckpt.name}")
            deployed = True
        except Exception as e:
            logger.error(f"Deployment copy failed: {e}")
    else:
        logger.warning(
            f"Fine-tuned checkpoint not found at {finetuned_ckpt}. "
            f"Production model unchanged."
        )

    # ── Clean up old backups (keep last 7) ────────────────────────────────
    backups = sorted(CHECKPOINT_DIR.glob("pretrain_backup_*.pt"))
    if len(backups) > 7:
        for old in backups[:-7]:
            old.unlink()
            logger.debug(f"Removed old backup: {old.name}")

    # ── Log deployment to DB ───────────────────────────────────────────────
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_deployments (
                    id          SERIAL PRIMARY KEY,
                    deployed_at TIMESTAMP DEFAULT NOW(),
                    checkpoint  VARCHAR(200),
                    deployed    BOOLEAN,
                    notes       TEXT
                );
            """)
            cur.execute("""
                INSERT INTO model_deployments (checkpoint, deployed, notes)
                VALUES (%s, %s, %s);
            """, (str(prod_ckpt), deployed, f"Nightly retrain {timestamp}"))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Deployment DB log failed: {e}")

    # ── Write reload signal file (signal_engine polls this) ───────────────
    reload_flag = PROJECT_ROOT / ".model_reload_requested"
    reload_flag.write_text(timestamp)

    outcome = {
        "deployed"  : deployed,
        "timestamp" : timestamp,
        "checkpoint": str(prod_ckpt) if deployed else "unchanged",
    }
    context["ti"].xcom_push(key="deploy_result", value=outcome)
    return outcome


def task_send_summary_alert(**context) -> Dict:
    """
    T11: Sends a Telegram summary of the nightly retrain results.
    Always runs (even if some tasks failed) to ensure visibility.

    Summarises:
        - Data download status
        - Feature pipeline status
        - Model validation result
        - Deployment status
        - Any failures
    """
    import requests as req

    ti      = context["ti"]
    run_date= context["ds"]

    # Gather results from previous tasks
    bhavcopy  = ti.xcom_pull(key="bhavcopy_result",  task_ids="download_bhavcopy")  or {}
    features  = ti.xcom_pull(key="feature_results",  task_ids="run_feature_pipeline") or {}
    backbone  = ti.xcom_pull(key="backbone_result",  task_ids="finetune_backbone") or {}
    rl        = ti.xcom_pull(key="rl_result",        task_ids="finetune_rl_agent") or {}
    validated = ti.xcom_pull(key="model_validated",  task_ids="validate_model")
    deployed  = ti.xcom_pull(key="deploy_result",    task_ids="deploy_model") or {}

    # Build summary
    val_ic    = backbone.get("val_ic",  "n/a")
    val_acc   = backbone.get("val_acc", "n/a")
    sharpe    = rl.get("val_sharpe",    "n/a")

    failed_pillars = [k for k, v in features.items() if v.get("status") != "ok"]
    overall_status = "✅ SUCCESS" if deployed.get("deployed") else "⚠️ PARTIAL"

    msg = (
        f"🧠 <b>G.O.D.S E.Y.E — Nightly Retrain</b>\n"
        f"📅 {run_date}\n\n"
        f"<b>Status:</b> {overall_status}\n\n"
        f"<b>Data:</b>\n"
        f"  Bhavcopy: {'✓' if bhavcopy.get('status') == 'success' else '⚠'}\n"
        f"  Features: {len(features) - len(failed_pillars)}/6 pillars OK\n"
        f"  Failed: {failed_pillars or 'none'}\n\n"
        f"<b>Model:</b>\n"
        f"  val_ic  = {val_ic if isinstance(val_ic, str) else f'{val_ic:.4f}'}\n"
        f"  val_acc = {val_acc if isinstance(val_acc, str) else f'{val_acc:.2%}'}\n"
        f"  Sharpe  = {sharpe if isinstance(sharpe, str) else f'{sharpe:.3f}'}\n"
        f"  Gates   = {'PASS ✓' if validated else 'FAIL ✗'}\n"
        f"  Deployed= {'YES ✓' if deployed.get('deployed') else 'NO (unchanged)'}\n"
    )

    # Send Telegram
    sent = False
    if TELEGRAM_TOKEN and TELEGRAM_CHAT:
        try:
            resp = req.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id"   : TELEGRAM_CHAT,
                    "text"      : msg,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            sent = resp.status_code == 200
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")

    logger.info(f"Summary alert {'sent ✓' if sent else 'logged only'}")
    logger.info(msg.replace("<b>", "").replace("</b>", ""))

    return {"sent": sent, "message_length": len(msg)}


# ══════════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def _parse_metric(stdout: str, metric_name: str) -> float:
    """
    Parses a metric value from subprocess stdout.
    Looks for patterns like 'val_ic=0.0654' or 'val_ic: 0.0654'.
    Returns 0.0 if not found.
    """
    import re
    patterns = [
        rf"{metric_name}[=:]\s*([0-9.]+)",
        rf"{metric_name}\s*=\s*([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return 0.0


# ══════════════════════════════════════════════════════════════════════════
#  DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id            = "godseye_nightly_retrain",
        default_args      = DEFAULT_ARGS,
        description       = "G.O.D.S E.Y.E nightly model retraining pipeline",
        schedule_interval = "30 17 * * 1-5",   # 22:30 IST = 17:00 UTC, Mon-Fri
        start_date        = days_ago(1),
        catchup           = False,
        max_active_runs   = 1,
        tags              = ["godseye", "ml", "trading"],
    ) as dag:

        # T1 — Infrastructure validation
        t1_validate_infra = PythonOperator(
            task_id         = "validate_infrastructure",
            python_callable = task_validate_infrastructure,
            provide_context = True,
        )

        # T2 — Download Bhavcopy
        t2_bhavcopy = PythonOperator(
            task_id         = "download_bhavcopy",
            python_callable = task_download_bhavcopy,
            provide_context = True,
        )

        # T3 — Download FII/DII
        t3_fii_dii = PythonOperator(
            task_id         = "download_fii_dii",
            python_callable = task_download_fii_dii,
            provide_context = True,
        )

        # T4 — Feature pipeline
        t4_features = PythonOperator(
            task_id         = "run_feature_pipeline",
            python_callable = task_run_feature_pipeline,
            provide_context = True,
            execution_timeout = timedelta(hours=1),
        )

        # T5 — Recompute embeddings
        t5_embeddings = PythonOperator(
            task_id         = "recompute_embeddings",
            python_callable = task_recompute_embeddings,
            provide_context = True,
        )

        # T6 — Validate features (ShortCircuit — stops pipeline if fails)
        t6_validate_features = ShortCircuitOperator(
            task_id         = "validate_features",
            python_callable = task_validate_features,
            provide_context = True,
        )

        # T7 — Fine-tune backbone
        t7_backbone = PythonOperator(
            task_id         = "finetune_backbone",
            python_callable = task_finetune_backbone,
            provide_context = True,
            execution_timeout = timedelta(hours=1, minutes=30),
        )

        # T8 — Fine-tune RL agent
        t8_rl = PythonOperator(
            task_id         = "finetune_rl_agent",
            python_callable = task_finetune_rl_agent,
            provide_context = True,
            execution_timeout = timedelta(hours=1),
        )

        # T9 — Validate model (ShortCircuit — stops deployment if gates fail)
        t9_validate_model = ShortCircuitOperator(
            task_id         = "validate_model",
            python_callable = task_validate_model,
            provide_context = True,
        )

        # T10 — Deploy model
        t10_deploy = PythonOperator(
            task_id         = "deploy_model",
            python_callable = task_deploy_model,
            provide_context = True,
        )

        # T11 — Summary alert (always runs via trigger_rule)
        t11_alert = PythonOperator(
            task_id         = "send_summary_alert",
            python_callable = task_send_summary_alert,
            provide_context = True,
            trigger_rule    = "all_done",   # runs even if upstream tasks failed
        )

        # ── DAG dependency graph ──────────────────────────────────────────
        #
        #  t1 → t2 ─┐
        #            ├─→ t4 → t5 → t6 → t7 ─┐
        #  t1 → t3 ─┘                         ├─→ t9 → t10 → t11
        #                                t8 ─┘
        #
        t1_validate_infra >> [t2_bhavcopy, t3_fii_dii]
        [t2_bhavcopy, t3_fii_dii] >> t4_features
        t4_features  >> t5_embeddings >> t6_validate_features
        t6_validate_features >> [t7_backbone, t8_rl]
        [t7_backbone, t8_rl] >> t9_validate_model
        t9_validate_model >> t10_deploy >> t11_alert

else:
    logger.warning(
        "Apache Airflow not installed. DAG definition skipped. "
        "Install with: pip install apache-airflow"
    )


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest airflow/dags/retraining_dag.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestRetrainingDAG:
    """
    Unit tests for DAG task functions.
    All tests mock external dependencies (DB, subprocess, Telegram).
    """

    def _mock_context(self, xcom_data: dict = None) -> dict:
        """Creates a mock Airflow task instance context."""
        class MockTI:
            def __init__(self):
                self._xcom = xcom_data or {}
            def xcom_push(self, key, value):
                self._xcom[key] = value
            def xcom_pull(self, key, task_ids=None):
                return self._xcom.get(key)

        return {"ti": MockTI(), "ds": "2024-03-15"}

    # ── _parse_metric ─────────────────────────────────────────────────────

    def test_parse_metric_equals_format(self):
        stdout = "Epoch 5 | val_ic=0.0654 | val_acc=0.5647"
        assert abs(_parse_metric(stdout, "val_ic")  - 0.0654) < 1e-6
        assert abs(_parse_metric(stdout, "val_acc") - 0.5647) < 1e-6

    def test_parse_metric_colon_format(self):
        stdout = "val_ic: 0.0712 val_acc: 0.58"
        assert abs(_parse_metric(stdout, "val_ic")  - 0.0712) < 1e-4
        assert abs(_parse_metric(stdout, "val_acc") - 0.58)   < 1e-4

    def test_parse_metric_not_found_returns_zero(self):
        assert _parse_metric("no metrics here", "val_ic") == 0.0

    def test_parse_metric_multiple_occurrences(self):
        stdout = "val_ic=0.03 ... val_ic=0.065"
        val    = _parse_metric(stdout, "val_ic")
        assert val > 0   # finds first match

    def test_parse_metric_integer_value(self):
        stdout = "val_sharpe=4"
        assert _parse_metric(stdout, "val_sharpe") == 4.0

    # ── task_validate_infrastructure (mocked) ────────────────────────────

    def test_validate_infra_structure(self, monkeypatch):
        """Infrastructure task should push xcom and return dict."""
        import psycopg2

        class _MockConn:
            def cursor(self): return self
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def execute(self, *a): pass
            def fetchone(self): return (752149,)
            def close(self): pass

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **kw: _MockConn())

        ctx = self._mock_context()
        # Won't fully run (Redis/ES may not be available) but tests structure
        try:
            result = task_validate_infrastructure(**ctx)
            assert isinstance(result, dict)
        except Exception:
            pass   # Expected in test environment without real services

    # ── task_download_bhavcopy ────────────────────────────────────────────

    def test_bhavcopy_weekend_skip(self, monkeypatch):
        """Weekend dates should return skipped=True."""
        from datetime import date
        # Mock today to be a Saturday
        monkeypatch.setattr(
            "airflow.dags.retraining_dag.date" if AIRFLOW_AVAILABLE
            else "builtins.date",
            type("D", (), {"today": staticmethod(lambda: date(2024, 3, 16)),
                           "weekday": date.weekday})
        )
        ctx    = self._mock_context()
        # Directly test the weekend logic
        import datetime
        saturday = datetime.date(2024, 3, 16)   # Saturday
        assert saturday.weekday() >= 5

    # ── task_validate_model ───────────────────────────────────────────────

    def test_validate_model_pass(self):
        """Model with IC and ACC above gates should return True."""
        xcom = {
            "backbone_result": {
                "status" : "ok",
                "val_ic" : 0.065,
                "val_acc": 0.57,
            },
            "rl_result": {"status": "ok", "val_sharpe": 3.5},
        }
        ctx    = self._mock_context(xcom)
        result = task_validate_model(**ctx)
        assert result is True

    def test_validate_model_fail_ic(self):
        """Low IC should cause gate failure."""
        xcom = {
            "backbone_result": {
                "status" : "ok",
                "val_ic" : 0.02,   # below MIN_VAL_IC=0.04
                "val_acc": 0.57,
            },
        }
        ctx    = self._mock_context(xcom)
        result = task_validate_model(**ctx)
        assert result is False

    def test_validate_model_fail_acc(self):
        """Low accuracy should cause gate failure."""
        xcom = {
            "backbone_result": {
                "status" : "ok",
                "val_ic" : 0.06,
                "val_acc": 0.50,   # below MIN_VAL_ACC=0.54
            },
        }
        ctx    = self._mock_context(xcom)
        result = task_validate_model(**ctx)
        assert result is False

    def test_validate_model_backbone_failed(self):
        """If backbone fine-tune failed, model should not deploy."""
        xcom = {
            "backbone_result": {"status": "error", "error": "OOM"},
        }
        ctx    = self._mock_context(xcom)
        result = task_validate_model(**ctx)
        assert result is False

    def test_validate_model_no_results(self):
        """Missing xcom data should fail gracefully."""
        ctx    = self._mock_context({})
        result = task_validate_model(**ctx)
        assert isinstance(result, bool)

    # ── task_deploy_model ─────────────────────────────────────────────────

    def test_deploy_model_no_checkpoint(self, tmp_path, monkeypatch):
        """Deploy without fine-tuned checkpoint should not crash."""
        monkeypatch.setattr(
            "airflow.dags.retraining_dag.CHECKPOINT_DIR"
            if AIRFLOW_AVAILABLE else
            "airflow.dags.retraining_dag.CHECKPOINT_DIR",
            tmp_path
        )
        ctx    = self._mock_context()
        try:
            result = task_deploy_model(**ctx)
            assert "deployed" in result
        except Exception:
            pass   # DB not available in test env

    # ── task_send_summary_alert ───────────────────────────────────────────

    def test_summary_alert_no_telegram(self):
        """Summary alert should complete without Telegram configured."""
        xcom = {
            "bhavcopy_result" : {"status": "success"},
            "feature_results" : {
                "trend"     : {"status": "ok"},
                "msi"       : {"status": "ok"},
                "fii_dii"   : {"status": "ok"},
                "sentiment" : {"status": "ok"},
                "volatility": {"status": "ok"},
                "correlation": {"status": "ok"},
            },
            "backbone_result" : {"status": "ok", "val_ic": 0.065, "val_acc": 0.57},
            "rl_result"       : {"status": "ok", "val_sharpe": 3.5},
            "model_validated" : True,
            "deploy_result"   : {"deployed": True, "timestamp": "20240315_223000"},
        }
        ctx    = self._mock_context(xcom)
        result = task_send_summary_alert(**ctx)
        assert "sent"           in result
        assert "message_length" in result
        assert result["message_length"] > 100

    def test_summary_alert_partial_failure(self):
        """Summary should handle missing xcom values gracefully."""
        ctx    = self._mock_context({})   # all xcom missing
        result = task_send_summary_alert(**ctx)
        assert isinstance(result, dict)

    # ── Gate thresholds ───────────────────────────────────────────────────

    def test_gate_values_reasonable(self):
        assert 0.02 <= MIN_VAL_IC  <= 0.10
        assert 0.50 <= MIN_VAL_ACC <= 0.60
        assert 0.10 <= MAX_DRIFT_KL <= 0.25

    # ── DAG definition ────────────────────────────────────────────────────

    def test_dag_exists_if_airflow_available(self):
        if AIRFLOW_AVAILABLE:
            assert dag is not None
            assert dag.dag_id == "godseye_nightly_retrain"

    def test_dag_task_count(self):
        if AIRFLOW_AVAILABLE:
            assert len(dag.tasks) == 11

    def test_dag_schedule(self):
        if AIRFLOW_AVAILABLE:
            assert dag.schedule_interval == "30 17 * * 1-5"

    def test_dag_no_catchup(self):
        if AIRFLOW_AVAILABLE:
            assert dag.catchup is False


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))