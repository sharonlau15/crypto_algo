# QF623 Crypto Algo Trading System

**SMU QF623 Group Project** — Multi-strategy algorithmic trading on Binance  
**Architecture**: 10 strategies → walk-forward backtest → seasonality selector → live execution

---

## Project structure

```
crypto_algo/
├── main.py                        ← Master entrypoint
├── requirements.txt
├── config/
│   ├── settings.py                ← All parameters (universe, risk limits, API)
│   └── client.py                  ← Dual Binance client (real data / testnet orders)
├── data/
│   ├── ingestion.py               ← OHLCV, funding rates, Fear & Greed
│   └── cache/                     ← Auto-generated Parquet cache
├── strategies/
│   ├── base.py                    ← Abstract base class
│   └── alpha.py                   ← All 10 strategies
├── backtest/
│   └── engine.py                  ← Walk-forward backtester (T+1/T+2 rule)
├── portfolio/
│   └── optimizer.py               ← Max-Sharpe optimizer (scipy SLSQP)
├── seasonality/
│   └── analyzer.py                ← Regime + seasonality + strategy selector
├── execution/
│   └── live_engine.py             ← Zero-touch live trading (APScheduler)
├── attribution/
│   └── factor_model.py            ← OLS factor regression + hedging analysis
├── utils/
│   ├── logger.py                  ← Loguru setup
│   └── reporting.py               ← Console tables + CSV/JSON output
├── results/                       ← Auto-generated output files
└── logs/                          ← Auto-generated log files
```

---

## Setup

```bash
# 1. Clone / download this folder
cd crypto_algo

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure your Binance.env file is at the path in config/settings.py
#    Required keys:
#      BINANCE_API_KEY
#      BINANCE_API_SECRET
#      BINANCE_TESTNET_API_KEY    (from https://testnet.binance.vision/)
#      BINANCE_TESTNET_API_SECRET
```

---

## Usage

```bash
# Run full backtest (all 10 strategies, seasonality, attribution)
python main.py --mode backtest

# Force re-download (ignore cache)
python main.py --mode backtest --no-cache

# Start zero-touch live trading on testnet (blocking — use tmux/screen)
python main.py --mode live

# Backtest + immediately go live
python main.py --mode full

# Print current live positions and NAV
python main.py --mode report
```

---

## 10 Alpha Strategies

| # | Strategy | Signal type | Key parameter |
|---|----------|-------------|---------------|
| 1 | Momentum | Time-series 12-1M | lookback_long=252 |
| 2 | Mean Reversion | Z-score | zscore_window=20 |
| 3 | Risk Parity | Inverse vol weights | lookback=63 |
| 4 | Cross-Sectional Mom | Rank-based | lookback=20 |
| 5 | Vol Breakout | ATR channels | atr_period=14 |
| 6 | Pairs Trading | BTC/ETH cointegration | entry_z=2.0 |
| 7 | ML Signal | LightGBM walk-forward | train_window=252 |
| 8 | Macro Rotation | BTC 200-day MA regime | — |
| 9 | Carry | Funding rate | lookback=7 |
| 10 | Sentiment | Fear & Greed index | fear_threshold=30 |

---

## Execution model (T+1/T+2 rule)

```
Day T (06:00 UTC):  Fetch data → compute signals → optimize weights
Day T+1 (market):  Orders placed at open (T+1 close in backtest)
Day T+1 → T+2:     Portfolio returns measured
```
This matches the project spec exactly: *"portfolio returns calculated from T+1 to T+2"*.

---

## Key outputs (in `results/`)

| File | Content |
|------|---------|
| `strategy_metrics.json` | Sharpe, CAGR, drawdown per strategy |
| `portfolio_returns.csv` | Daily return series for all strategies |
| `cumulative_returns.csv` | Cumulative NAV curves |
| `monthly_seasonality.csv` | Average monthly Sharpe per strategy |
| `regime_performance.csv` | Sharpe by bull/bear/sideways regime |
| `attribution_report.csv` | Factor betas, alpha, R², IR |
| `live_state.json` | Live positions, cash, NAV history |

---

## Risk controls

- `PAPER_TRADING = True` — hardcoded default, testnet only
- `MAX_POSITION_SIZE = 0.40` — no single token > 40% of NAV
- `MAX_WEIGHT_SUM = 1.00` — gross leverage capped at 100%
- `MIN_ANNUALIZED_VOL = 0.03` — portfolio vol ≥ 3% (project spec)
- Minimum order notional: $11 USDT (Binance floor + buffer)
- Covariance: 1-year (252 bars) of daily returns

---

## Notes for project presentation

- All backtest returns are **net of transaction costs** (10 bps + 5 bps slippage)
- No look-ahead bias: signals are shifted by 1 day before portfolio construction
- ML strategy uses strict walk-forward cross-validation (no future data in train set)
- Seasonality analysis requires minimum 24 observations per regime/month bucket
