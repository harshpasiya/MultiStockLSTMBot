# ZODIC OMEGA  
Risk-First Multi-Stock LSTM Trading Engine

---

## Overview

ZODIC OMEGA is a risk-driven, portfolio-aware trading system built around a shared LSTM model and strict portfolio constraints.  
It is designed to scale from single-stock research to multi-stock live trading without overtrading, data leakage, or architectural fragility.

Unlike signal-heavy systems, ZODIC OMEGA treats the model as a weak forecaster and places decision authority in the risk and portfolio layer.

---

## Design Philosophy

ZODIC OMEGA follows five core principles:

1. Risk over signals  
2. Portfolio over individual trades  
3. Shared intelligence  
4. Selective participation  
5. Reproducibility and discipline  

---


## Core Pipeline

### 1. Data Collection
`data_collection.py`  
Fetches historical OHLCV data for all stocks defined in `config/universe.yaml`.

---

### 2. Feature Engineering
`feature_engineering.py`  
Applies a fixed, shared feature schema across all stocks.

Feature order is immutable and defined in `config/features.yaml`.

---

### 3. Sequence Creation
`sequence_creation.py`  
Transforms features into rolling sequences suitable for LSTM training.

Sequences from all stocks are stacked into a single dataset.

---

### 4. Model Architecture
`build_lstm_model.py`  
Defines the shared LSTM architecture only.

No data access  
No training logic  
No risk logic  

---

### 5. Model Training
`train_model.py`  
Trains the shared LSTM model on all stocks simultaneously.

Outputs model weights and scaler artifacts.

---

### 6. Evaluation
`evaluate_model.py`  
Evaluates predictive behavior (loss, directional accuracy, stability).

No trading logic is present here.

---

### 7. Signal Generation
`generate_signals.py`  
Runs inference across all stocks and produces candidate opportunities.

Signals are filtered using:
- confidence thresholds  
- percentile ranking  
- expected return constraints  

---

### 8. Portfolio Backtesting
`backtest_strategy.py`  
Simulates portfolio-level execution with:
- capital allocation  
- max concurrent positions  
- max trades per month  
- drawdown limits  
- holding period constraints  

All risk rules are loaded from `config/risk.yaml`.

---

### 9. Live Execution
`live_signal.py`  
Production entry point.

- Loads the latest model and scaler  
- Runs inference on the most recent data  
- Applies portfolio and risk constraints  
- Outputs only actionable decisions:
  - BUY  
  - SELL  
  - or no output (HOLD)

Designed to be run once per candle/day via scheduler or cron.

---

## Configuration Files

### `config/universe.yaml`
Defines the stock universe (symbols, sectors, optional metadata).

---

### `config/features.yaml`
Defines:
- feature names  
- feature order  
- normalization rules  

This file must never change once the model is trained.

---

### `config/risk.yaml`
The single source of truth for all risk management:

- risk per trade  
- max concurrent positions  
- max trades per month  
- stop loss / take profit  
- max drawdown  
- holding period limits  

No risk parameters are hard-coded anywhere else.

---

## Model Strategy

- One shared LSTM model  
- Trained on all stocks simultaneously  
- No per-stock models by default  
- No probability mapping assumptions  
- Signals are ranked, not blindly traded  

Per-stock specialization is considered only after sustained live performance.

---

## Execution Philosophy

- Silence is expected and desired  
- The system will often produce no trades  
- Capital preservation has priority over activity  
- Overtrading is explicitly prevented by design  

---

## Environment Requirements

- Python 3.10 or 3.11  
- TensorFlow / Keras (supported versions)  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  

Exact versions should be frozen once moving to live deployment.

---

## Status

This repository represents the transition from single-stock experimentation to multi-stock, risk-governed portfolio trading.

The single-stock LSTM pipeline is considered complete and serves as a validated reference implementation.

---

## Disclaimer

This project is for research and educational purposes only.  
It does not constitute financial advice or a recommendation to trade.