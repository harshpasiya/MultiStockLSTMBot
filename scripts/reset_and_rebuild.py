"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Universe Reset & Rebuild                        ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : scripts/reset_and_rebuild.py                           ║
║                                                                          ║
║  What this script does:                                                  ║
║    1. Wipes all training data for the old 500-stock universe             ║
║    2. Sets up the new focused 47-stock universe                          ║
║    3. Verifies which symbols exist in daily_ohlcv                        ║
║    4. Rebuilds universe.yaml                                             ║
║    5. Clears old feature tables                                          ║
║    6. Clears old embeddings                                              ║
║    7. Deletes old checkpoints                                            ║
║    8. Prints the exact commands to run next                              ║
║                                                                          ║
║  Usage:                                                                  ║
║    python -m scripts.reset_and_rebuild                                   ║
║    python -m scripts.reset_and_rebuild --dry-run   (preview only)       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import yaml
import shutil
import argparse
import psycopg2

from pathlib import Path
from loguru  import logger
from dotenv  import load_dotenv

load_dotenv()

DB_URL   = os.getenv("TIMESCALE_URL", "postgresql://godseye_user:godseye_pass@localhost:5433/godseye")
ROOT_DIR = Path(__file__).parent.parent

# ── Focused 47-stock universe ──────────────────────────────────────────────
FOCUSED_UNIVERSE = [
    # Power & Energy (10)
    "TATAPOWER", "JSWENERGY", "ADANIPOWER", "SUZLON", "NHPC",
    "SJVN", "TORNTPOWER", "CESC", "RPOWER", "INOXWIND",

    # Defence & Shipbuilding (5)
    "BEL", "BEML", "COCHINSHIP", "MAZDOCK", "GRSE",

    # Railways & Infrastructure (5)
    "RVNL", "IRCON", "TITAGARH", "RAILTEL", "IRFC",

    # Capital Goods & Electricals (8)
    "CUMMINSIND", "SIEMENS", "ABB", "CGPOWER", "KEI",
    "POLYCAB", "APARINDS", "THERMAX",

    # Automobiles & Auto Ancillaries (5)
    "TATAMOTORS", "ASHOKLEY", "BHARATFORG", "MOTHERSON", "TVSMOTOR",

    # NBFCs & PSU Banks (6)
    "AUBANK", "FEDERALBNK", "BANKBARODA", "PNB", "RECLTD", "PFC", "CHOLAFIN",

    # Chemicals (3)
    "DEEPAKNTR", "AARTIIND", "SRF",

    # Pharmaceuticals (4)
    "LUPIN", "AUROPHARMA", "LAURUSLABS", "GLENMARK",
]

# Sector mapping for RC-10 (sector concentration rule)
SECTOR_MAP = {
    # Power
    "TATAPOWER": "Power", "JSWENERGY": "Power", "ADANIPOWER": "Power",
    "SUZLON": "Power", "NHPC": "Power", "SJVN": "Power",
    "TORNTPOWER": "Power", "CESC": "Power", "RPOWER": "Power",
    "INOXWIND": "Power",
    # Defence
    "BEL": "Defence", "BEML": "Defence", "COCHINSHIP": "Defence",
    "MAZDOCK": "Defence", "GRSE": "Defence",
    # Railways
    "RVNL": "Railways", "IRCON": "Railways", "TITAGARH": "Railways",
    "RAILTEL": "Railways", "IRFC": "Railways",
    # Capital Goods
    "CUMMINSIND": "CapGoods", "SIEMENS": "CapGoods", "ABB": "CapGoods",
    "CGPOWER": "CapGoods", "KEI": "CapGoods", "POLYCAB": "CapGoods",
    "APARINDS": "CapGoods", "THERMAX": "CapGoods",
    # Auto
    "TATAMOTORS": "Auto", "ASHOKLEY": "Auto", "BHARATFORG": "Auto",
    "MOTHERSON": "Auto", "TVSMOTOR": "Auto",
    # NBFC/Banks
    "AUBANK": "NBFC", "FEDERALBNK": "NBFC", "BANKBARODA": "NBFC",
    "PNB": "NBFC", "RECLTD": "NBFC", "PFC": "NBFC", "CHOLAFIN": "NBFC",
    # Chemicals
    "DEEPAKNTR": "Chemicals", "AARTIIND": "Chemicals", "SRF": "Chemicals",
    # Pharma
    "LUPIN": "Pharma", "AUROPHARMA": "Pharma",
    "LAURUSLABS": "Pharma", "GLENMARK": "Pharma",
}


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1: VERIFY SYMBOLS IN DB
# ══════════════════════════════════════════════════════════════════════════

def verify_symbols(dry_run: bool = False) -> list[str]:
    """
    Checks which symbols from FOCUSED_UNIVERSE exist in daily_ohlcv.
    Returns the confirmed available symbols.
    """
    logger.info("Step 1: Verifying symbols in daily_ohlcv...")

    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM daily_ohlcv;")
        db_symbols = {row[0] for row in cur.fetchall()}
    conn.close()

    available   = []
    missing     = []

    for sym in FOCUSED_UNIVERSE:
        if sym in db_symbols:
            available.append(sym)
        else:
            missing.append(sym)

    logger.info(f"  ✓ Available in DB : {len(available)}/{len(FOCUSED_UNIVERSE)}")

    if missing:
        logger.warning(f"  ✗ Missing from DB : {missing}")
        logger.warning(
            "  Missing symbols need to be downloaded via NSE Bhavcopy.\n"
            "  They will be excluded from this run.\n"
            "  Run data/ingestion/nse_bhavcopy.py to backfill them."
        )

    return available


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2: WIPE OLD FEATURE TABLES
# ══════════════════════════════════════════════════════════════════════════

FEATURE_TABLES = [
    "features_trend",
    "features_msi",
    "features_fii_dii",
    "features_sentiment",
    "features_volatility",
    "features_correlation",
    "features_fused",
    "backbone_embeddings",
    "correlation_matrix",
]

def wipe_feature_tables(available_symbols: list[str], dry_run: bool = False):
    """
    Deletes all feature data for ALL symbols (full wipe).
    We wipe everything because the model is being retrained from scratch
    on a different universe — stale features from old symbols would corrupt training.
    """
    logger.info("Step 2: Wiping old feature tables...")

    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            for table in FEATURE_TABLES:
                # Check if table exists first
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    );
                """, (table,))
                exists = cur.fetchone()[0]

                if not exists:
                    logger.info(f"  {table}: does not exist — skipping")
                    continue

                if dry_run:
                    cur.execute(f"SELECT COUNT(*) FROM {table};")
                    count = cur.fetchone()[0]
                    logger.info(f"  [DRY RUN] Would delete {count:,} rows from {table}")
                else:
                    cur.execute(f"TRUNCATE TABLE {table};")
                    logger.success(f"  ✓ {table}: truncated")

        if not dry_run:
            conn.commit()
            logger.success("All feature tables wiped.")
        else:
            conn.rollback()

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to wipe tables: {e}")
        raise
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3: FILTER daily_ohlcv TO NEW UNIVERSE
# ══════════════════════════════════════════════════════════════════════════

def filter_ohlcv_to_universe(available_symbols: list[str], dry_run: bool = False):
    """
    Removes all OHLCV data for symbols NOT in the focused universe.
    This significantly reduces the dataset size and speeds up data loading.

    Before: ~752K rows, 499 symbols
    After : ~75K rows, 47 symbols (estimated)
    """
    logger.info("Step 3: Filtering daily_ohlcv to focused universe...")

    placeholders = ",".join(["%s"] * len(available_symbols))
    conn = psycopg2.connect(DB_URL)

    try:
        with conn.cursor() as cur:
            # Count rows to be deleted
            cur.execute(f"""
                SELECT COUNT(*) FROM daily_ohlcv
                WHERE symbol NOT IN ({placeholders});
            """, available_symbols)
            to_delete = cur.fetchone()[0]

            cur.execute(f"""
                SELECT COUNT(*) FROM daily_ohlcv
                WHERE symbol IN ({placeholders});
            """, available_symbols)
            to_keep = cur.fetchone()[0]

            logger.info(f"  Rows to keep   : {to_keep:,}")
            logger.info(f"  Rows to delete : {to_delete:,}")

            if dry_run:
                logger.info("  [DRY RUN] Would delete rows for non-universe symbols")
            else:
                cur.execute(f"""
                    DELETE FROM daily_ohlcv
                    WHERE symbol NOT IN ({placeholders});
                """, available_symbols)
                deleted = cur.rowcount
                logger.success(f"  ✓ Deleted {deleted:,} rows for non-universe symbols")

        if not dry_run:
            conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to filter OHLCV: {e}")
        raise
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4: REBUILD universe.yaml
# ══════════════════════════════════════════════════════════════════════════

def rebuild_universe_yaml(available_symbols: list[str], dry_run: bool = False):
    """
    Rewrites config/universe.yaml with the focused universe.
    Also saves sector_map for RC-10 use.
    """
    logger.info("Step 4: Rebuilding config/universe.yaml...")

    config_dir = ROOT_DIR / "config"
    config_dir.mkdir(exist_ok=True)

    universe_path = config_dir / "universe.yaml"
    sector_path   = config_dir / "sector_map.yaml"

    # Filter sector map to available symbols only
    filtered_sector = {
        sym: SECTOR_MAP[sym]
        for sym in available_symbols
        if sym in SECTOR_MAP
    }

    universe_data = {
        "nifty500"  : sorted(available_symbols),   # key name kept for compatibility
        "n_stocks"  : len(available_symbols),
        "description": "Focused 47-stock universe: Power, Defence, Railways, CapGoods, Auto, NBFC, Chemicals, Pharma",
        "version"   : "2.0",
    }

    sector_data = {
        "sector_map" : filtered_sector,
        "sectors"    : list(set(filtered_sector.values())),
    }

    if dry_run:
        logger.info(f"  [DRY RUN] Would write {len(available_symbols)} symbols to universe.yaml")
        logger.info(f"  [DRY RUN] Sectors: {list(set(filtered_sector.values()))}")
    else:
        with open(universe_path, "w") as f:
            yaml.dump(universe_data, f, default_flow_style=False, sort_keys=False)
        with open(sector_path, "w") as f:
            yaml.dump(sector_data, f, default_flow_style=False, sort_keys=False)
        logger.success(f"  ✓ universe.yaml written: {len(available_symbols)} symbols")
        logger.success(f"  ✓ sector_map.yaml written: {len(set(filtered_sector.values()))} sectors")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 5: DELETE OLD CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════

def delete_checkpoints(dry_run: bool = False):
    """
    Deletes all old training checkpoints.
    These were trained on 500 stocks — useless for the new universe.
    """
    logger.info("Step 5: Deleting old checkpoints...")

    checkpoint_dir = ROOT_DIR / "checkpoints"
    if not checkpoint_dir.exists():
        logger.info("  No checkpoints directory found.")
        return

    deleted = []
    for f in checkpoint_dir.iterdir():
        if f.suffix in (".pt", ".zip"):
            if dry_run:
                logger.info(f"  [DRY RUN] Would delete: {f.name}")
            else:
                f.unlink()
                deleted.append(f.name)

    if not dry_run and deleted:
        logger.success(f"  ✓ Deleted {len(deleted)} checkpoint files: {deleted}")
    elif not deleted:
        logger.info("  No checkpoint files found.")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 6: CLEAR LOGS
# ══════════════════════════════════════════════════════════════════════════

def clear_logs(dry_run: bool = False):
    """Clears TensorBoard logs and progress CSVs from old training runs."""
    logger.info("Step 6: Clearing old training logs...")

    log_dirs = [
        ROOT_DIR / "logs" / "swing_rl",
        ROOT_DIR / "logs" / "walk_forward",
        ROOT_DIR / "logs" / "evaluate",
    ]

    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        if dry_run:
            logger.info(f"  [DRY RUN] Would clear: {log_dir}")
        else:
            shutil.rmtree(log_dir)
            log_dir.mkdir(parents=True)
            logger.success(f"  ✓ Cleared: {log_dir}")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 7: UPDATE model_config.yaml
# ══════════════════════════════════════════════════════════════════════════

def update_model_config(n_stocks: int, dry_run: bool = False):
    """
    Updates config/model_config.yaml with the new N_STOCKS value
    and faster training settings for the focused universe.
    """
    logger.info("Step 7: Updating config/model_config.yaml...")

    config_path = ROOT_DIR / "config" / "model_config.yaml"

    config = {
        # Universe
        "n_stocks"            : n_stocks,
        "embedding_dim"       : 128,
        "seq_len"             : 60,

        # Observation space
        "obs_dim"             : n_stocks * 128 + 8,

        # Training — tuned for focused universe
        "total_timesteps"     : 5_000_000,    # 5M instead of 10M (smaller universe = faster learning)
        "n_envs"              : 8,
        "n_steps"             : 2048,
        "batch_size"          : 256,
        "n_epochs"            : 10,
        "learning_rate_start" : 3e-4,
        "learning_rate_end"   : 1e-5,
        "gamma"               : 0.99,
        "gae_lambda"          : 0.95,
        "clip_range"          : 0.2,
        "vf_coef"             : 0.5,
        "ent_coef"            : 0.05,          # higher entropy for better exploration
        "max_grad_norm"       : 0.5,
        "patience_steps"      : 1_500_000,     # 1.5M patience (faster feedback)
        "eval_freq"           : 50_000,        # evaluate every 50K (more frequent)
        "eval_episodes"       : 20,
        "early_stop_sharpe"   : 1.8,
        "backbone_unfreeze"   : 1_000_000,     # unfreeze at 1M (not 2M)

        # RL heads
        "swing_tp_pct"        : 0.04,
        "swing_sl_pct"        : 0.015,
        "intraday_tp_pct"     : 0.025,
        "intraday_sl_pct"     : 0.008,

        # Risk Constitution
        "max_positions"       : 4,
        "max_trades_month"    : 15,
        "max_drawdown_kill"   : 0.12,
    }

    if dry_run:
        logger.info(f"  [DRY RUN] Would write model_config.yaml with N_STOCKS={n_stocks}")
    else:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.success(f"  ✓ model_config.yaml updated: N_STOCKS={n_stocks}, obs_dim={config['obs_dim']}")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 8: PRINT NEXT COMMANDS
# ══════════════════════════════════════════════════════════════════════════

def print_next_steps(available_symbols: list[str]):
    """Prints the exact sequence of commands to run after this script."""
    n = len(available_symbols)
    obs_dim = n * 128 + 8

    print("\n" + "═" * 65)
    print("  RESET COMPLETE — RUN THESE COMMANDS IN ORDER")
    print("═" * 65)
    print()
    print("# 1. Update N_STOCKS in train_swing_rl.py:")
    print(f"#    Change N_STOCKS = 50 → N_STOCKS = {n}")
    print(f"#    (obs_dim will be {n} × 128 + 8 = {obs_dim})")
    print()
    print("# 2. Update N_STOCKS in precompute_embeddings.py:")
    print(f"#    Same change: N_STOCKS = {n}")
    print()
    print("# 3. Pre-compute embeddings for focused universe (~5 mins):")
    print("     python -m training.precompute_embeddings")
    print()
    print("# 4. Run Phase 1 feature engineering on focused universe:")
    print("     python -m features.trend --mode all")
    print("     python -m features.msi --mode all")
    print("     python -m features.volatility --mode all")
    print("     python -m features.correlation --mode all")
    print("     python -m features.fusion --mode all")
    print()
    print("# 5. Re-run backbone pre-training on focused universe:")
    print("     python -m training.pretrain_backbone")
    print()
    print("# 6. Start RL training (fresh, no --resume):")
    print("     python -m training.train_swing_rl")
    print()
    print(f"  Universe: {n} stocks across 8 sectors")
    print(f"  Expected training time: 6–10 hours (vs 20h before)")
    print(f"  Expected val Sharpe: >1.2 within 2M steps")
    print("═" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E — Universe Reset & Rebuild"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview all changes without executing them"
    )
    parser.add_argument(
        "--skip-ohlcv-filter", action="store_true",
        help="Skip deleting non-universe OHLCV rows (faster, keeps all data)"
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.warning("DRY RUN MODE — no changes will be made")

    logger.info("=" * 60)
    logger.info("G.O.D.S E.Y.E — Universe Reset & Rebuild")
    logger.info(f"New universe: {len(FOCUSED_UNIVERSE)} focused stocks")
    logger.info("=" * 60)

    # Step 1: Verify symbols
    available = verify_symbols(args.dry_run)

    if len(available) < 20:
        logger.error(
            f"Only {len(available)} symbols found in DB. "
            f"Need at least 20. Run NSE Bhavcopy backfill first."
        )
        sys.exit(1)

    # Step 2: Wipe feature tables
    wipe_feature_tables(available, args.dry_run)

    # Step 3: Filter OHLCV (optional — keeps DB clean)
    if not args.skip_ohlcv_filter:
        filter_ohlcv_to_universe(available, args.dry_run)
    else:
        logger.info("Step 3: Skipped OHLCV filter (--skip-ohlcv-filter)")

    # Step 4: Rebuild universe.yaml + sector_map.yaml
    rebuild_universe_yaml(available, args.dry_run)

    # Step 5: Delete old checkpoints
    delete_checkpoints(args.dry_run)

    # Step 6: Clear logs
    clear_logs(args.dry_run)

    # Step 7: Update model_config.yaml
    update_model_config(len(available), args.dry_run)

    # Step 8: Print next steps
    if not args.dry_run:
        print_next_steps(available)
    else:
        logger.info("\nDRY RUN complete. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()