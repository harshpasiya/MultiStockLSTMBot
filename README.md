<div align="center">

```
 ██████╗  ██████╗ ██████╗ ███████╗    ███████╗██╗   ██╗███████╗
██╔════╝ ██╔═══██╗██╔══██╗██╔════╝    ██╔════╝╚██╗ ██╔╝██╔════╝
██║  ███╗██║   ██║██║  ██║███████╗    █████╗   ╚████╔╝ █████╗
██║   ██║██║   ██║██║  ██║╚════██║    ██╔══╝    ╚██╔╝  ██╔══╝
╚██████╔╝╚██████╔╝██████╔╝███████║    ███████╗   ██║   ███████╗
 ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝    ╚══════╝   ╚═╝   ╚══════╝
```

### **Generative Observation & Decision System for Equity Intelligence**
*An autonomous AI trading intelligence for the Indian financial markets*

---

![Python](https://img.shields.io/badge/Python-3.11+-0D1B3E?style=for-the-badge&logo=python&logoColor=C9A84C)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-0D1B3E?style=for-the-badge&logo=pytorch&logoColor=C9A84C)
![Status](https://img.shields.io/badge/Status-In_Development-C9A84C?style=for-the-badge)
![Market](https://img.shields.io/badge/Market-NSE_|_BSE-0D1B3E?style=for-the-badge)
![Universe](https://img.shields.io/badge/Universe-Nifty_500-0D1B3E?style=for-the-badge&logoColor=C9A84C)
![License](https://img.shields.io/badge/License-Private-C9A84C?style=for-the-badge)

</div>

---

## What is G.O.D.S E.Y.E?

G.O.D.S E.Y.E is a **fully autonomous, self-learning AI trading system** built for the Indian equity markets (NSE/BSE). It observes the entire Nifty 500 universe continuously, learns from 6 independent intelligence dimensions, and generates high-conviction buy/hold/sell signals for both swing and intraday trading — without being explicitly told what to do after deployment.

It is not a screener. It is not a rule-based bot. It is a continuously self-improving AI fund manager that runs 24/7, manages risk automatically, and gets smarter every single trading day through its nightly retraining pipeline.

> *"The eye that never sleeps, the mind that never forgets, the discipline that never breaks."*

---

## Performance Targets

| Metric | Target |
|--------|--------|
| 📈 CAGR | ≥ 45% |
| 🎯 Win Rate | 58 – 65% |
| 📉 Max Drawdown | ≤ 12% |
| ⚖️ Sharpe Ratio | ≥ 1.8 |
| 💰 Profit Factor | ≥ 2.0 |
| 🔄 Max Trades / Month | 15 |
| 📊 Max Active Positions | 4 |

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                      │
│  OHLCV │ Options OI │ FII/DII Flow │ News/NLP │ Microstructure│
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  6 LEARNING PILLARS                          │
│                                                              │
│  [1] Trend      [2] MSI (Vol-Weighted OB/OS)                │
│  [3] FII/DII    [4] Sentiment (FinBERT NLP)                 │
│  [5] Volatility [6] Correlation & Inter-stock Influence      │
│                + Adaptive Kelly Position Sizing              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│               SHARED AI BACKBONE                             │
│                                                              │
│   BiLSTM Encoder  →  Transformer Encoder  →  128d Embedding │
│   (Temporal memory)  (Cross-asset attention) (Fused repr.)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
┌─────────▼────────┐     ┌─────────▼────────┐
│  SWING RL HEAD   │     │ INTRADAY RL HEAD  │
│  PPO · Daily bars│     │ PPO · 5m/15m bars │
│  TP 4% / SL 1.5% │     │ TP 2.5% / SL 0.8% │
└─────────┬────────┘     └─────────┬────────┘
          └────────────┬────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              RISK CONSTITUTION (Hard Constraints)            │
│         10 unbreakable rules the RL agent cannot override    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│           SIGNAL OUTPUT & BROKER EXECUTION                   │
│         Zerodha Kite Connect │ Upstox │ Angel Broking        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│         NIGHTLY SELF-RETRAINING PIPELINE (Airflow)           │
│    Trade outcomes → retrain all 6 pillars → deploy at 00:30  │
└─────────────────────────────────────────────────────────────┘
```

---

## The 6 Learning Pillars

### Pillar 1 — Trend Analysis
Learns direction, strength, and maturity of price trends across multiple timeframes using EMA Ribbon, ADX, Supertrend, VWAP deviation, and Ichimoku Cloud. Output: trend score from **-1.0 (strong downtrend)** to **+1.0 (strong uptrend)**.

### Pillar 2 — MSI (Market Sentiment Index)
A proprietary volume-weighted overbought/oversold indicator. Combines Volume-Weighted RSI, Money Flow Index, NSE Delivery % Momentum, and Put-Call Ratio into a single composite score **[0–100]**. Unlike standard RSI, MSI is weighted by the actual money flowing in and out.

### Pillar 3 — FII/DII Flow Analysis
Operates at the **market level**, not the stock level. Processes both provisional (4 PM) and final (6 PM) FII/DII data from NSE daily, plus FII index futures OI and options net gamma. Outputs a **Market Direction Signal (MDS)** from -3 (strong bearish) to +3 (strong bullish) every morning before 9:15 AM.

### Pillar 4 — Sentiment Analysis (NLP)
Uses **FinBERT** fine-tuned on an Indian financial news corpus. Operates at two levels simultaneously — market-wide fear/greed index and per-stock sentiment score. Sources include BSE filings, Economic Times, Moneycontrol, and sector news feeds. Updates every 30 minutes during market hours.

### Pillar 5 — Volatility Analysis
Tracks current and predicted volatility regime using ATR, Historical Volatility, Implied Volatility Rank (IVR), GARCH(1,1) forecasting, and real-time India VIX. Volatility directly controls position sizing and TP/SL placement — high volatility automatically widens stops and reduces size.

### Pillar 6 — Correlation & Inter-stock Influence
Prevents concentration risk and identifies alpha opportunities. Uses a 60-day dynamic correlation matrix, PCA on the Nifty 500 return matrix, lead-lag relationship detection, and a Graph Neural Network (GNN) to model complex non-linear inter-stock influence (e.g., Reliance → downstream suppliers).

---

## Risk Management

### Position Level
| Parameter | Swing | Intraday |
|-----------|-------|----------|
| Take Profit | 4.0% | 2.5% |
| Stop Loss | 1.5% | 0.8% |
| Trailing Stop Activation | After +2% move | After +1.5% move |
| Max Hold Duration | 15 trading days | End of day (forced close) |
| ATR Adjustment | Widens by 0.5× ATR when ATR > 2% | Widens by 0.3× ATR when ATR > 1.5% |

### The Risk Constitution — 10 Hard Rules
Rules the RL agent **cannot override under any circumstance:**

| Rule | Trigger | Response |
|------|---------|----------|
| RC-01 | Drawdown > 12% from peak | All trading halted; human review required |
| RC-02 | Single position loss > 2% of portfolio | Immediate forced exit |
| RC-03 | 2+ positions with correlation > 0.75 | New signal rejected |
| RC-04 | Earnings within 2 trading days | No new entry in that stock |
| RC-05 | Stock within 5% of circuit limit | No entry; tighten trailing stop |
| RC-06 | India VIX +8% intraday OR Nifty -3% in 30 min | All intraday signals blocked |
| RC-07 | Stock avg daily volume < ₹5 crore | Stock excluded from universe |
| RC-08 | 4 positions already open | No new signals acted upon |
| RC-09 | FII sells > ₹3000 crore provisional | MDS forced to -3; all longs blocked |
| RC-10 | 2 positions from same sector | No third position in that sector |

---

## Project Structure

```
MultiStockLSTMBot/               ← GitHub repo (G.O.D.S E.Y.E internally)
│
├── config/                      ← Hyperparameters, risk thresholds, universe
├── data/
│   ├── ingestion/               ← Kite feed, NSE Bhavcopy, options, FII/DII, news
│   ├── pipeline/                ← Corporate actions, normalization, gap handling
│   └── validators/              ← Data quality & staleness checks
├── features/                    ← All 6 learning pillars + fusion module
├── models/                      ← BiLSTM, Transformer, Backbone, RL agents
├── environment/                 ← Gymnasium env, Risk Constitution, Kelly sizer
├── training/                    ← Pretraining, RL training, walk-forward backtester
├── execution/                   ← Signal engine, Kite/Upstox executors, failover
├── monitoring/                  ← Grafana, Prometheus metrics, drift detector
├── airflow/dags/                ← Nightly retraining DAG (10:30 PM IST)
├── api/                         ← FastAPI signal broadcast + subscriber management
└── tests/                       ← Unit + integration tests for every module
```

---

## Development Roadmap

| Phase | Name | Duration | Gate Criterion |
|-------|------|----------|----------------|
| **0** | Data Infrastructure | Weeks 1–3 | All 10 sources live; < 0.1% data gaps |
| **1** | Feature Engineering | Weeks 4–6 | All 6 pillars validated; IC > 0.02 |
| **2** | AI Backbone Pre-training | Weeks 7–11 | Validation Sharpe > 0.8 |
| **3** | RL Agent Training | Weeks 12–18 | Simulated CAGR > 35%; drawdown < 15% |
| **4** | Walk-Forward Backtesting | Weeks 19–22 | 45%+ CAGR; ≤ 12% DD; Sharpe ≥ 1.8 |
| **5** | Paper Trading | Weeks 23–28 | 6-week track record within 10% of backtest |
| **6** | Live Trading (Tranches) | Weeks 29–40 | Each tranche passes 8-week no-breach test |
| **7** | Multi-Account & Subscribers | Week 41+ | SEBI compliance review complete |

---

## Tech Stack

```
Deep Learning     →  PyTorch 2.1+
RL Framework      →  Stable-Baselines3 (PPO)
NLP / Sentiment   →  HuggingFace Transformers (FinBERT)
Technical Anal.   →  pandas-ta, TA-Lib
Volatility        →  arch (GARCH)
Backtesting       →  VectorBT
Broker APIs       →  Zerodha Kite Connect, Upstox SDK
Time-Series DB    →  TimescaleDB (PostgreSQL)
Search / Text     →  Elasticsearch
Cache             →  Redis
Orchestration     →  Apache Airflow
Model Tracking    →  MLflow
Monitoring        →  Grafana + Prometheus
API Server        →  FastAPI + Uvicorn
Containerization  →  Docker + Docker Compose
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker + Docker Compose (for TimescaleDB, Redis, Elasticsearch)
- NVIDIA GPU recommended for training (RTX 3090 or better)
- Zerodha Kite Connect API credentials

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOURUSERNAME/MultiStockLSTMBot.git
cd MultiStockLSTMBot

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start infrastructure services
docker-compose up -d

# 5. Configure environment
cp .env.example .env
# Edit .env with your API keys and DB credentials

# 6. Verify setup
python -m pytest tests/ -v
```

### Environment Variables (`.env`)

```env
# Broker APIs
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
UPSTOX_API_KEY=your_upstox_api_key

# Database
TIMESCALE_URL=postgresql://user:password@localhost:5432/godseye
REDIS_URL=redis://localhost:6379
ELASTICSEARCH_URL=http://localhost:9200

# News & Data
NEWS_API_KEY=your_news_api_key

# Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Current Development Status

- [x] Project architecture finalized
- [x] Full technical roadmap documented
- [x] File structure created
- [ ] Phase 0: Data infrastructure (in progress)
- [ ] Phase 1: Feature engineering pipeline
- [ ] Phase 2: AI backbone pre-training
- [ ] Phase 3: RL agent training
- [ ] Phase 4: Walk-forward backtesting
- [ ] Phase 5: Paper trading
- [ ] Phase 6: Live trading
- [ ] Phase 7: Subscriber platform

---

## Important Disclaimers

> **RISK WARNING:** Algorithmic trading involves substantial risk of loss. Past performance of any model in backtesting does not guarantee future results. This system is for personal research and development purposes.

> **REGULATORY NOTE:** Providing automated trading signals to third-party subscribers for a fee is a SEBI-regulated activity in India. SEBI Research Analyst (RA) or Investment Adviser (IA) registration is required before onboarding paying subscribers.

> **NOT FINANCIAL ADVICE:** Nothing in this repository constitutes financial advice. Trade at your own risk.

---

## License

This project is **private and proprietary**. All rights reserved.

---

<div align="center">

*Built for the Indian markets. Designed to compound.*

**G.O.D.S E.Y.E** — *The eye that sees everything. The mind that learns forever.*

</div>