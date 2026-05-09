# QF623 / QF635 Crypto Algorithmic Trading System

**SMU Quantitative Finance Group Project**  
Multi-strategy algorithmic trading on Binance — backtest, live futures execution, and interactive dashboard.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Project Structure](#3-project-structure)
4. [Setup & Installation](#4-setup--installation)
5. [API Keys & Environment](#5-api-keys--environment)
6. [Running the System](#6-running-the-system)
7. [The 10 Alpha Strategies](#7-the-10-alpha-strategies)
8. [How Signals Become Orders](#8-how-signals-become-orders)
9. [Backtest Engine](#9-backtest-engine)
10. [Portfolio Optimizer](#10-portfolio-optimizer)
11. [Seasonality & Regime Selector](#11-seasonality--regime-selector)
12. [Live Feedback Loop](#12-live-feedback-loop)
13. [Live Engine — Two Loops](#13-live-engine--two-loops)
14. [Risk Controls](#14-risk-controls)
15. [NAV & P&L Accounting](#15-nav--pl-accounting)
16. [Dashboard](#16-dashboard)
17. [Performance Attribution](#17-performance-attribution)
18. [Data Layer](#18-data-layer)
19. [Configuration Reference](#19-configuration-reference)
20. [Output Files](#20-output-files)
21. [Deployment (Server)](#21-deployment-server)
22. [Common Questions & Troubleshooting](#22-common-questions--troubleshooting)

---

## 1. System Overview

This system runs **10 independent alpha strategies** simultaneously. Each strategy produces a signal matrix (conviction in `[-1, +1]` per token per day). A meta-layer called the **Seasonality/Regime Selector** picks the 1–2 best-performing strategies for the current market regime and blends their signals. The blended signal drives a **Max-Sharpe portfolio optimizer** that outputs target weights. A live engine translates weight changes into real Binance Futures Demo orders — including real short positions.

In parallel, all 10 strategies run as **hypothetical paper portfolios** whose live P&L feeds back into the strategy selector — so the system learns which strategies are working *right now* and tilts toward them automatically.

```
Market Data → 10 Strategies → Signals → Selector (Regime + Live P&L) → Optimizer → Orders
                                ↑                                         |
                                └────── Hypothetical P&L feedback ────────┘
```

**Key design decisions:**
- **Futures Demo Trading** — `PAPER_TRADING = True` is hardcoded as the default; virtual funds on `testnet.binancefuture.com`, real market prices
- **Long and short** — futures perpetuals allow real short positions; live engine uses `long_short=True`
- **Signal-driven, not clock-driven** — orders only fire when signal weight delta exceeds `REBALANCE_THRESHOLD`
- **Zero look-ahead** — signals are shifted 1 day before entering the optimizer; ML model trained on strictly past data
- **Binance-authoritative balances** — wallet balance and positions reconciled from Binance Futures every cycle; `initial_nav` stamped once from the real account equity on first run

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                │
│   --mode backtest | live | full | report                        │
└───────────────┬─────────────────────────────────────────────────┘
                │
        ┌───────▼──────────────────────────────────────────┐
        │           BACKTEST PIPELINE                       │
        │                                                   │
        │  data/ingestion.py  ─────► Parquet cache          │
        │        │                                          │
        │        ▼                                          │
        │  strategies/alpha.py (×10) ─► signals_dict       │
        │        │                                          │
        │        ▼                                          │
        │  backtest/engine.py  ─────► BacktestResult ×10   │
        │        │                                          │
        │        ▼                                          │
        │  seasonality/analyzer.py ─► SeasonalityAnalyzer  │
        │        │                                          │
        │        ▼                                          │
        │  attribution/factor_model.py ─► OLS attribution  │
        │        │                                          │
        │        ▼                                          │
        │  utils/reporting.py ──────► results/*.csv/json    │
        └───────────────────────────────────────────────────┘
                │
                ▼ (--mode live or full)
        ┌───────────────────────────────────────────────────┐
        │           LIVE ENGINE (execution/live_engine.py)  │
        │                                                   │
        │  APScheduler                                      │
        │  ├── price_monitor_job (every 60s)                │
        │  │     Poll futures prices → check stop/TP → exit │
        │  └── signal_rebalance_job (every 1 min)           │
        │        Fetch data → recompute signals →           │
        │        reconcile Binance Futures account →        │
        │        update hypotheticals → select strategy →   │
        │        optimize weights → compare to live →       │
        │        execute futures orders if delta > threshold │
        │                                                   │
        │  State: results/live_state.json (atomic writes)   │
        └───────────────────────────────────────────────────┘
                │
                ▼
        ┌───────────────────────────────────────────────────┐
        │           DASHBOARD (dashboard.py)                │
        │                                                   │
        │  streamlit run dashboard.py                       │
        │  ├── Backtest Analysis tab                        │
        │  ├── Live Trading (Futures) tab                   │
        │  │     Reads live_state.json (30s TTL)            │
        │  │     Fetches live positions from Binance        │
        │  │     Futures API (wallet balance + positions)   │
        │  └── Strategy Monitor tab                         │
        └───────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
crypto_algo/
├── main.py                        ← Master entrypoint (argparse)
├── dashboard.py                   ← Streamlit dashboard
├── debug_futures.py               ← Futures API connectivity diagnostic
├── requirements.txt
├── Binance.env                    ← API keys (gitignored)
│
├── config/
│   ├── settings.py                ← ALL tuneable parameters
│   └── client.py                  ← Dual client: real data / demo futures orders
│
├── data/
│   ├── ingestion.py               ← OHLCV, funding rates, Fear & Greed fetch
│   └── cache/                     ← Auto-generated Parquet files (6h TTL)
│
├── strategies/
│   ├── base.py                    ← BaseStrategy abstract class
│   └── alpha.py                   ← All 10 strategies
│
├── backtest/
│   └── engine.py                  ← Walk-forward backtester (T+1/T+2)
│
├── portfolio/
│   ├── optimizer.py               ← Max-Sharpe SLSQP optimizer
│   └── risk_manager.py            ← Position-level risk helpers
│
├── seasonality/
│   └── analyzer.py                ← Regime + seasonality + strategy selector
│
├── execution/
│   └── live_engine.py             ← APScheduler live trading engine
│
├── attribution/
│   └── factor_model.py            ← OLS factor regression
│
├── utils/
│   ├── logger.py                  ← Loguru setup
│   └── reporting.py               ← Console tables + CSV/JSON output
│
├── results/                       ← Auto-generated (gitignored)
│   ├── live_state.json            ← Live positions, wallet balance, NAV history
│   ├── strategy_metrics.json
│   ├── portfolio_returns.csv
│   └── ...
│
└── logs/                          ← Auto-generated log files
```

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10+
- A Binance account with a real API key (for market data)
- A Binance Demo Trading account with futures-enabled API key (for paper trading orders)

### Local Setup

```bash
# 1. Navigate into the project
cd crypto_algo

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

### Server Setup (Ubuntu/Debian)

```bash
# Install Python if needed
sudo apt update && sudo apt install python3-pip python3-venv -y

# Clone / copy the project
cd /opt && git clone https://github.com/sharonlau15/crypto_algo.git
cd crypto_algo

# Create venv and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create API key file (see Section 5)
nano Binance.env

# Run in background with tmux
tmux new-session -d -s crypto "python3 main.py --mode full"
```

---

## 5. API Keys & Environment

Create a file called `Binance.env` in the project root (`crypto_algo/Binance.env`). This file is gitignored — never commit it.

```env
BINANCE_API_KEY=your_real_api_key_here
BINANCE_API_SECRET=your_real_api_secret_here
BINANCE_DEMO_API_KEY=your_demo_trading_key_here
BINANCE_DEMO_API_SECRET=your_demo_trading_secret_here
```

**Why two sets of keys?**
| Client | Purpose | Key source |
|--------|---------|-----------|
| Real Binance | Market data (OHLCV, tickers, funding rates) — read-only | binance.com → API Management |
| Demo Trading | Futures order placement, account balance queries — virtual money | binance.com → Demo Trading → API Management |

**Important — Futures endpoint:**
- Binance Demo Trading Futures routes through `testnet.binancefuture.com`, **not** `fapi.binance.com`
- The demo client uses `testnet=True` in python-binance, which sets the futures URL correctly
- Market data (OHLCV, prices) always comes from the real client at `api.binance.com` — unaffected
- The demo API key must have **Enable Futures** ticked in Binance API settings

**How the dual client works (`config/client.py`):**
```python
get_client(for_trading=False)  # → real Binance (market data, read-only)
get_client(for_trading=True)   # → demo futures (orders, account balance)
```

**Verifying connectivity:**
```bash
source venv/bin/activate && python3 debug_futures.py
```
This checks both endpoints and prints your wallet balance. If `futures_account` fails, check that "Enable Futures" is ticked on the demo API key.

---

## 6. Running the System

```bash
# Run full backtest only (no live trading)
python3 main.py --mode backtest

# Force re-download of all market data (ignore parquet cache)
python3 main.py --mode backtest --no-cache

# Start live futures trading (runs backtest first, then goes live)
python3 main.py --mode live

# Backtest + immediately launch live engine
python3 main.py --mode full

# Fire one live rebalance immediately on startup (don't wait for scheduler)
python3 main.py --mode live --run-now

# Print current positions and NAV from state file
python3 main.py --mode report

# Launch the interactive dashboard (separate terminal)
streamlit run dashboard.py
```

### What `--mode full` does step by step:
1. Fetches OHLCV data for all 12 tokens (from cache if fresh)
2. Fetches funding rates and Fear & Greed index
3. Generates signals for all 10 strategies
4. Runs walk-forward backtest for each strategy (IS and OOS split)
5. Computes regime + monthly seasonality scores
6. Runs factor attribution
7. Saves all results to `results/`
8. Sets leverage for all universe symbols via Binance Futures API
9. Starts the APScheduler live engine (blocking — keeps running until Ctrl+C)

---

## 7. The 10 Alpha Strategies

All strategies live in `strategies/alpha.py` and inherit from `BaseStrategy`. Each outputs a signal DataFrame of shape `(dates × tokens)` with values in `[-1, +1]`.

- `+1.0` = maximum bullish conviction (long)
- `-1.0` = maximum bearish conviction (short)
- `0.0` = neutral / no position

### Strategy 1 — Momentum (`momentum`)
**What it does:** Classic 12-1 month time-series momentum. Computes the 12-month return minus the 1-month return for each token. Tokens with the highest scores get a `+1` signal (long); lowest scores get `-1` (short).

**Why it works:** Price momentum is one of the most documented factors across asset classes (Jegadeesh & Titman 1993). In crypto, momentum is particularly strong due to narrative-driven retail flows.

**Key parameters:**
- `lookback_long = 252` — 12-month lookback (in daily bars)
- `lookback_short = 21` — skip the last 1 month (avoids short-term reversal)
- `top_n = 4` — long top 4 tokens
- `bottom_n = 4` — short bottom 4 tokens

**Signal type:** Binary ±1 (not continuous)

---

### Strategy 2 — Mean Reversion (`mean_reversion`)
**What it does:** Computes a rolling z-score of each token's price relative to its own 20-day moving average. Tokens that have fallen sharply (low z-score) get a positive (long) signal; tokens that spiked (high z-score) get a short signal.

**Why it works:** Short-term crypto prices overreact to news events and then mean-revert. This is especially pronounced on 3–20 day windows.

**Key parameters:**
- `zscore_window = 20` — rolling mean/std window
- `entry_z = 2.0` — z-score of ±2 maps to signal ±1
- `exit_z = 0.5` — not used directly (continuous signal handles exit automatically)

**Signal formula:** `signal = -zscore / (entry_z × 3)` clipped to `[-1, +1]`. A z-score of -2 produces a signal of ~+0.33; a z-score of -6 produces +1.

---

### Strategy 3 — Risk Parity (`risk_parity`)
**What it does:** Weights tokens inversely proportional to their recent volatility so each token contributes the same amount of risk. Low-vol tokens get higher weights; high-vol tokens get lower weights. Long-only, always fully invested.

**Why it works:** Diversifies risk rather than capital. Outperforms equal-weight during high cross-sectional vol dispersion because it doesn't over-concentrate in the most volatile (often trending) tokens.

**Key parameters:**
- `lookback = 63` — 3-month rolling vol window

**Implementation note:** If a token has zero vol (e.g., stale/delisted like MATICUSDT post-migration), its inverse-vol would be infinite. The code handles this with `replace([inf, -inf], nan)` so the stale token is excluded rather than poisoning the entire portfolio.

---

### Strategy 4 — Cross-Sectional Momentum (`cross_sectional_momentum`)
**What it does:** Ranks all 12 tokens by their 20-day return each day. The rank is rescaled to `[-1, +1]` continuously — the top-ranked token gets `+1`, the bottom gets `-1`.

**Why it works:** Relative momentum within the same asset class removes the systematic market beta (you're long relative winners and short relative losers). Unlike time-series momentum, this strategy is always market-neutral in signal space.

**Key parameters:**
- `lookback = 20` — 20-day return ranking window
- `rank_method = "min"` — ties broken by minimum rank

---

### Strategy 5 — Volatility Breakout (`vol_breakout`)
**What it does:** ATR (Average True Range) channel breakout. If today's close is above yesterday's close + (ATR × multiplier), the signal is positive (long breakout). If below yesterday's close - (ATR × multiplier), it is negative (short breakdown).

**Why it works:** Volatility compression followed by expansion signals the start of a new directional trend. Common in technical trading; effective in crypto where vol clustering is pronounced.

**Key parameters:**
- `atr_period = 14` — rolling window for ATR calculation
- `atr_multiplier = 2.0` — ATR band half-width multiplier
- `lookback = 20`

**Signal formula:** `(close - prev_close) / (atr_mult × ATR)` clipped to `[-1, +1]`. Continuously graded — a large breakout gets a signal near ±1.

---

### Strategy 6 — Pairs Trading (`pairs_trading`)
**What it does:** BTC/ETH cointegration spread trade. Estimates the hedge ratio β via OLS regression on a rolling 60-day window: `spread = log(BTC) - β × log(ETH)`. When the spread is wide (high z-score), it shorts BTC and longs ETH. When spread is narrow (low z-score), the opposite.

**Why it works:** BTC and ETH share systematic crypto risk factors (macro, regulation, sentiment). Their idiosyncratic spread tends to mean-revert. The rolling OLS hedge ratio adapts to structural changes in the relationship.

**Key parameters:**
- `pair = ("BTCUSDT", "ETHUSDT")`
- `lookback = 60` — rolling OLS window
- `entry_z = 2.0` — z-score of ±2 maps to signal ±1
- `exit_z = 0.0` — signal returns to 0 when spread returns to mean

**Note:** The cointegration p-value gate (`coint_pvalue = 0.05`) in config is currently not enforced — the signal is always live. This was intentional to avoid the strategy going silent during regime breaks.

---

### Strategy 7 — ML Signal (`ml_signal`)
**What it does:** LightGBM gradient boosting classifier trained on lagged return features. Target variable = sign of next-day return (1 = up, 0 = down/flat). Trained separately for each token. Signal = predicted probability of up-move, rescaled to `[-1, +1]`.

**Features used:**
- Lagged returns at [1, 3, 5, 10, 21] days
- 10-day and 21-day rolling volatility
- 21-day rolling skewness
- Vol ratio (10d vol / 21d vol) — captures regime changes

**Strict no-look-ahead (walk-forward):** The model retrains every 30 days. For each prediction at index `i`, the model is trained on `[i - train_window, i-1]` only — the last row of the training window is excluded from both X_train and y_train to prevent target leakage. Predictions are always fully out-of-sample.

**Key parameters:**
- `feature_lookbacks = [1, 3, 5, 10, 21]`
- `train_window = 180` — 6-month rolling training window
- `n_estimators = 200`
- `max_depth = 4`
- `learning_rate = 0.05`

---

### Strategy 8 — Macro Rotation (`macro_rotation`)
**What it does:** Uses BTC's 20-day return as a macro risk-on/risk-off indicator. If BTC's recent return is positive (risk-on), all tokens get positive signals proportional to their own recent return. If BTC is negative (risk-off), signals are suppressed or reversed.

**Why it works:** BTC is the dominant systematic factor in crypto. Its trend strongly influences altcoin performance. This strategy is effectively a market-regime filter that scales exposure based on macro conditions.

**Key parameters:**
- `risk_on_threshold = 0.0` — positive 20d BTC return = risk-on
- `lookback = 20` — BTC return lookback
- `btc_proxy = "BTCUSDT"`

---

### Strategy 9 — Carry (`carry`)
**What it does:** Uses Binance perpetual futures 8-hour funding rates as a carry signal. Tokens with consistently positive funding rates (longs paying shorts) are in demand — this is bullish. Tokens with negative funding rates are under short pressure — bearish.

**Why it works:** Perpetual funding rates are the crypto equivalent of the carry factor in FX. Positive funding = market is net long, reflecting bullish sentiment. Negative funding = market is net short.

**Key parameters:**
- `lookback = 7` — 7-day average funding rate
- `top_n = 4` — long top 4 by funding carry

**Data source:** `data/ingestion.py → get_universe_funding_rates()` which calls Binance `/fapi/v1/fundingRate`. Falls back gracefully if the endpoint is unavailable.

---

### Strategy 10 — Sentiment (`sentiment`)
**What it does:** Uses the Alternative.me Crypto Fear & Greed Index (0–100). When the index is above 60 (greed), the market is risk-on — all tokens get a positive signal. When below 30 (fear), it uses a contrarian long signal. Between 30–60 = neutral.

**Why it works:** The Fear & Greed Index captures collective market sentiment. Extreme fear often precedes reversals (contrarian); greed during established trends can reinforce momentum.

**Key parameters:**
- `greed_threshold = 60` — above this → risk-on long bias
- `fear_threshold = 30` — below this → contrarian long
- `lookback = 7` — trailing average of the index

**Data source:** `https://api.alternative.me/fng/` — free, no authentication required.

---

## 8. How Signals Become Orders

This is the full pipeline from a strategy's output to an actual Binance Futures order.

### Step 1 — Signal generation
Each strategy outputs a float in `[-1, +1]` per token. This is conviction strength — not just direction.

### Step 2 — Strategy blending (`seasonality/analyzer.py → blend_signals`)
The top-2 strategies selected by the regime/seasonality selector are blended with proportional weights:
```
blended_signal = Σ (strategy_weight × strategy_signal) / total_weight
```
The blended signal is clipped to `[-1, +1]`.

### Step 3 — Weight conversion (`_signal_to_weights`)
Signals are converted to portfolio weights with `long_short=True` (futures supports real shorts):
- Top 6 positive signals → long positions (weights > 0)
- Top 6 negative signals → short positions (weights < 0)
- Long and short each get up to 50% of gross exposure when both sides are present
- Any token exceeding `MAX_POSITION_SIZE = 20%` is capped; the excess is redistributed

Example:
```
Signals:  BTC=0.8, SOL=0.4, ETH=-0.6
Long budget: 50% → BTC: 33%, SOL: 17%
Short budget: 50% → ETH: -50% (capped at -20%, excess redistributed)
```

### Step 4 — Price fetch (live futures ticker)
```python
client.futures_symbol_ticker()  # returns live mark prices for all symbols
```
Real-time prices from Binance Futures at the moment of rebalance — not the stale daily close from the parquet cache.

### Step 5 — Delta computation (`execute_rebalance`)
```
current_w[sym]  = qty[sym] × price / NAV   (signed: negative for shorts)
delta_weight    = target_weight - current_weight
delta_usdt      = delta_weight × NAV
qty             = delta_usdt / live_price
qty             = rounded down to Binance futures lot-size step
```

### Step 6 — Order placement
All orders are `MARKET` type via `futures_create_order()`. Exit orders use `reduceOnly=True` to prevent accidental position reversal.

Order flow:
- `delta_w > 0` → BUY (open/add long, or close short)
- `delta_w < 0` → SELL (open/add short, or close long)
- `|delta_usdt| < MIN_ORDER_USDT ($11)` → skip (Binance minimum notional)

### Step 7 — State update
After each fill, the state is updated:
- `positions[sym]` updated with new signed quantity (positive=long, negative=short)
- `cash_usdt` updated with realized P&L on closing trades
- `position_entries[sym]` recorded with entry price, date, peak/trough price
- Trade logged to `trade_log` with P&L

---

## 9. Backtest Engine

**File:** `backtest/engine.py`

### T+1/T+2 execution rule (matches project spec exactly)
```
Day T     → Signal computed from close prices up to day T
Day T+1   → Portfolio rebalanced at close of T+1 (signal shifted 1 day)
T+1→T+2   → Returns measured over the next day
```

This is implemented by shifting signals by 1 day before multiplying by returns:
```python
weights_lagged = weights.shift(1)       # T+1 execution
period_returns = (weights_lagged * returns)  # T+1 → T+2 returns
```

### Train / Test Split
```
In-sample:     BACKTEST_START (2023-01-01) → BACKTEST_TEST_START (2025-01-01)
Out-of-sample: BACKTEST_TEST_START (2025-01-01) → today
```

All metrics (Sharpe, CAGR, etc.) are reported for **full period**, **in-sample**, and **out-of-sample** separately. The IS/OOS comparison is shown in the dashboard under Backtest Analysis. Out-of-sample performance is the honest evaluation — in-sample results may reflect overfitting.

### Transaction costs
```
Net return = gross return - transaction_cost
cost = |Δweight| × TRANSACTION_COST_BP / 10000
```
- `TRANSACTION_COST_BP = 10` — 10 bps round-trip (Binance maker/taker ~5 bps each side)
- `SLIPPAGE_BP = 5` — additional 5 bps assumed slippage per trade (used in backtest only)

### Metrics computed for each strategy
| Metric | Formula |
|--------|---------|
| Sharpe | `(mean_ret × 365) / (std_ret × √365)` |
| Sortino | `(mean_ret × 365) / (downside_std × √365)` |
| Max Drawdown | `min((NAV - rolling_max) / rolling_max)` |
| Calmar | `CAGR / |max_drawdown|` |
| CAGR | `(final_NAV / initial_NAV)^(1/years) - 1` |
| Win Rate | Fraction of days with positive return |
| Best/Worst Day P&L | In USDT from $10k hypothetical starting capital |

All annualization uses **365** (crypto markets never close).

### Walk-forward validation
The ML strategy uses a strict walk-forward design: the model retrains every 30 days and for each prediction at index `i`, trains only on `[i - train_window, i-1]`. The last row of the training window is excluded from both X and y to prevent target leakage. Other strategies are fully causal by construction (all lookbacks use `.shift()` to avoid touching future data).

---

## 10. Portfolio Optimizer

**File:** `portfolio/optimizer.py`

### Method
Max-Sharpe optimization using `scipy.optimize.minimize` with the SLSQP method. Strategy signals act as expected return proxies (`μ_i = signal_i`).

**Objective:** Maximize `Sharpe = μᵀw / √(wᵀΣw)`

**Constraints:**
- `Σ|wᵢ| ≤ 1.0` (gross leverage cap)
- `√(wᵀΣw) ≥ 0.03` (minimum annualized portfolio vol, project spec)
- `-0.20 ≤ wᵢ ≤ +0.20` (per-token cap)

**Covariance matrix:**
- Estimated from the last `RISK_LOOKBACK_DAYS = 180` days of daily returns
- Annualized by multiplying by 365
- Regularized with `+1e-6 × I` to avoid numerical singularity
- All `sqrt(w'Σw)` calls protected with `max(0.0, ...)` to avoid `sqrt(negative)` from floating-point rounding

**Fallback:** If SLSQP fails to converge, equal-weight among tokens with positive signals. The engine never goes fully flat due to optimizer failure.

---

## 11. Seasonality & Regime Selector

**File:** `seasonality/analyzer.py`

### Layer 1 — Market Regime Classification
BTC's 200-day MA + 30-day realized volatility determine the current regime:

| Condition | Regime |
|-----------|--------|
| BTC > 200-day MA AND vol < 75th percentile | `bull` |
| BTC < 200-day MA AND vol < 75th percentile | `bear` |
| vol > 75th percentile (regardless of price) | `sideways` |

### Layer 2 — Regime Performance Score
For each strategy × regime combination, compute the annualized Sharpe ratio from backtest returns. Only periods with ≥24 observations count (configurable via `SEASONALITY_MIN_PERIODS`).

### Layer 3 — Monthly Seasonality Score
For each strategy, compute the average monthly Sharpe across the backtest period. This captures calendar patterns (e.g., momentum works better in January, mean reversion in August).

### Layer 4 — Combined Backtest Score
```
combined_score = 0.70 × regime_sharpe + 0.30 × monthly_sharpe
```

### Layer 5 — Live Feedback Score (the adaptive layer)
As the live engine runs, each strategy maintains a **hypothetical paper portfolio** updated every minute. The selector blends in the live rolling Sharpe:

```
final_score = (1 - live_weight) × backtest_score + live_weight × live_sharpe_normalized
```

`live_weight` ramps from 0% → 80% over the first 2 days:
```python
live_weight = min(days_live / 2, 0.80)
```

**Why 2 days / 80%?** Crypto regimes change fast. A 14-day ramp (like traditional equity quant systems) is too slow. The 2-day window captures the current momentum while the 20% backtest floor prevents over-fitting to a single day's lucky noise.

**Why not 100% live?** Sharpe computed over 48 hours is noisy. A strategy that happens to be up 2% in the first 12 hours shouldn't completely displace a strategy with a superior 2-year backtest.

### Selection output
Top-2 strategies by final score, with blend weights proportional to their scores:
```
[("momentum", 0.6), ("cross_sectional_momentum", 0.4)]
```

---

## 12. Live Feedback Loop

### How all 10 strategies run simultaneously
Every time `signal_rebalance_job` fires (every 1 minute), it:
1. Fetches the latest prices
2. Updates a **hypothetical paper portfolio** for each of the 10 strategies
3. Each hypothetical tracks: NAV, weights, last prices, trade history, entry prices

### Hypothetical NAV update
```python
period_return = Σ prev_weight[sym] × (price_now - price_prev) / price_prev
new_nav = prev_nav × (1 + period_return)
```

Shorts are modeled correctly: negative weights produce negative returns when prices rise (good for shorts when market falls).

### Rolling Sharpe computation (48h window)
```python
recent_navs = nav_history[last_48h]
returns = recent_navs.pct_change().dropna()
rolling_sharpe = returns.mean() / returns.std()
```

### When a strategy "takes over"
If a strategy's live rolling Sharpe is significantly better than the backtest-selected strategies, `live_weight` (capped at 80%) shifts the blend. Over 2 days, the live score dominates and the system effectively switches to the better-performing strategy.

### NAV history cap
Nav history is capped at 2880 entries (2 days × 1440 minutes) to prevent the state file from growing unboundedly. Hypothetical trade history is capped at 500 entries per strategy.

---

## 13. Live Engine — Two Loops

**File:** `execution/live_engine.py`

### Loop 1: Price Monitor (`price_monitor_job`)
- **Frequency:** Every `PRICE_MONITOR_SECS = 60` seconds
- **Purpose:** Poll futures prices for open positions and immediately close any position that breaches a risk limit
- **Does NOT rebalance** — purely a safety mechanism

Checks in order (direction-aware for longs and shorts):
1. Hard stop loss: `signed_pct ≤ -STOP_LOSS_PCT`
2. Take profit: `signed_pct ≥ TAKE_PROFIT_PCT`
3. Trailing stop: for longs — drop from peak price; for shorts — rise from trough price

Also prints a heartbeat log every 30 seconds showing NAV, wallet balance, and per-position P&L.

### Loop 2: Signal Recompute + Conditional Rebalance (`signal_rebalance_job`)
- **Frequency:** Every `SIGNAL_RECOMPUTE_MINS = 1` minute
- **Purpose:** Recompute all signals on fresh data and rebalance IF the signal has changed materially

Steps:
1. Fetch latest OHLCV data (no cache — always fresh)
2. Recompute signals for all 10 strategies
3. Snapshot latest signals to state (dashboard reads these)
4. Fetch current futures prices
5. **Reconcile with Binance Futures** — sync positions (signed qty), wallet balance, stamp `initial_nav` on first cycle
6. Update all 10 hypothetical paper portfolios
7. Select active strategies (regime + live P&L)
8. Compute new target weights for live portfolio (long and short)
9. Compare new weights to actual current positions
10. If `Σ|Δweight| > REBALANCE_THRESHOLD (3%)` → execute futures orders; else skip

### Why signal-driven (not clock-driven)?
If BTC surges 10% at 3am, the system rebalances within 1 minute regardless of schedule. If the market is flat, no unnecessary trading occurs even when the scheduler fires.

### State file
`results/live_state.json` — written atomically via `.tmp` rename (POSIX atomic) so a crash mid-write never produces a corrupted file. Key fields:

```json
{
  "positions":        {"BTCUSDT": 0.012345, "ETHUSDT": -0.5, ...},
  "cash_usdt":        7234.56,
  "initial_nav":      10802.74,
  "nav_history":      [{"date": "...", "nav": 10234.56}, ...],
  "current_weights":  {"BTCUSDT": 0.20, "ETHUSDT": -0.15, ...},
  "position_entries": {"BTCUSDT": {"entry_price": 65000, "entry_date": "...", "peak_price": 66000}},
  "trade_log":        [{"time": "...", "symbol": "BTCUSDT", "side": "BUY", "qty": 0.012, ...}],
  "active_strategies": ["momentum", "cross_sectional_momentum"],
  "active_strategy_weights": {"momentum": 0.6, "cross_sectional_momentum": 0.4},
  "latest_signals":   {"momentum": {"BTCUSDT": 1.0, "ETHUSDT": -0.5, ...}, ...},
  "hypothetical":     {"momentum": {"nav": 10500, "weights": {...}, ...}, ...}
}
```

Note: `positions[sym]` is **signed** — positive = long, negative = short. `cash_usdt` = futures wallet balance (USDT), synced from Binance each cycle.

### Bootstrapping on fresh start (no state file)
When `live_state.json` doesn't exist, `load_state()` calls `_bootstrap_from_binance()`:
1. Reads actual signed position quantities from the Binance Futures account
2. Reads entry prices for open positions from Binance (`entryPrice` field)
3. Reads actual USDT wallet balance from the futures account
4. Computes `initial_nav = wallet_balance + totalUnrealizedProfit`
5. This allows the engine to pick up where it left off after a redeploy

### NAV guard
If NAV is zero or negative (unfunded futures wallet), the engine logs a CRITICAL message and calls `os._exit(1)` to stop the process immediately. Fund the Binance Futures Demo account and restart.

---

## 14. Risk Controls

### Portfolio-level
| Control | Value | Effect |
|---------|-------|--------|
| `PAPER_TRADING = True` | default | All orders go to Demo Futures, never live |
| `MAX_WEIGHT_SUM = 1.00` | 100% | Total gross leverage capped |
| `MAX_POSITION_SIZE = 0.20` | 20% | Single token cap (long or short) |
| `MAX_LIVE_POSITIONS = 6` | 6 | Max simultaneous positions per side |
| `MIN_ANNUALIZED_VOL = 0.03` | 3% | Portfolio vol floor (project spec) |
| `REBALANCE_THRESHOLD = 0.03` | 3% | Min weight delta to trigger orders |
| `FUTURES_LEVERAGE = 1` | 1x | Leverage set on all symbols at startup |

### Position-level
| Control | Value | Trigger |
|---------|-------|---------|
| `STOP_LOSS_PCT = 0.02` | 2% | Exit if signed P&L drops below -2% |
| `TAKE_PROFIT_PCT = 0.03` | 3% | Exit if signed P&L exceeds +3% |
| `TRAILING_STOP_PCT = 0.05` | 5% | Exit if price moves 5% against peak/trough |
| `USE_TRAILING_STOP = True` | on | Trailing stop is active |

### Execution safety
- Exit orders use `reduceOnly=True` — prevents an exit from accidentally reversing a position
- `MIN_ORDER_USDT = 11` — Binance minimum notional + buffer
- Futures LOT_SIZE step fetched from `futures_exchange_info()` — quantities rounded down to avoid precision errors
- Quantity computation: `qty = abs(delta_usdt) / price` then rounded to step size

---

## 15. NAV & P&L Accounting

### NAV calculation (futures)
```
NAV = wallet_balance + unrealized_PnL
unrealized_PnL = Σ qty[sym] × (current_price[sym] - entry_price[sym])
```

This works correctly for both longs (qty > 0) and shorts (qty < 0). `wallet_balance` is the USDT balance in the Binance Futures account, which already includes all realized P&L from closed trades.

### P&L tracking
- **Unrealized P&L** = `qty × (current_price - entry_price)` for each open position (signed correctly for longs/shorts)
- **Realized P&L** = `direction × qty_closed × (close_price - entry_price)` logged at each trade
- **Total P&L** = `current_NAV - initial_nav`

### `initial_nav`
Stamped once from the actual Binance Futures account equity (`wallet_balance + totalUnrealizedProfit`) on the first successful reconciliation cycle. **Never overwritten** after that — provides a stable P&L baseline even after restarts. Stored in `live_state.json`.

### Binance reconciliation
Every signal cycle, before any NAV or weight computation:
1. `client.futures_account()` fetches `walletBalance` (USDT) and `positionAmt` (signed qty) for all universe symbols
2. State is updated: `positions` ← signed Binance quantities; `cash_usdt` ← Binance wallet balance
3. Entry prices for positions that were opened outside the current session are seeded from Binance's `entryPrice` field
4. `initial_nav` is stamped if not yet set

### Profit reinvestment
All profits stay in the portfolio. As wallet balance grows from realized gains, the next rebalance sizes positions based on the higher NAV, automatically compounding returns.

---

## 16. Dashboard

**File:** `dashboard.py`  
**Command:** `streamlit run dashboard.py`

The dashboard has three main sections selectable from the sidebar.

### Backtest Analysis
- Strategy performance comparison (Sharpe, CAGR, Max Drawdown, Sortino, Calmar) — full period, IS, OOS
- Cumulative NAV curves for all strategies
- In-sample vs Out-of-sample comparison table
- Monthly seasonality heatmap (strategy × month)
- Regime performance heatmap (strategy × bull/bear/sideways)
- Factor attribution table (beta to BTC, ETH, vol, momentum, carry)
- P&L summary

### Live Trading (Futures)

**Active Strategy Banner** — shows which 1–2 strategies are currently active with blend percentages. Explained as: "70% Momentum + 30% Cross-Sectional Momentum" driven by backtest regime/seasonality score + 48h live rolling Sharpe.

**NAV Metrics** — five headline numbers:
- Futures NAV (wallet balance + unrealized PnL from Binance Futures API)
- Wallet Balance (USDT from Binance Futures — includes all realized PnL)
- Unrealized PnL (sum of open position mark-to-market)
- Open positions count
- Total P&L (vs `initial_nav` stamped from real Binance account on first cycle)

**NAV History Chart** — time series from `nav_history` with the opening balance baseline.

**Open Positions Table** — shows:
- Token, Side (LONG/SHORT), quantity, entry price, live price (from Binance Futures ticker), market value
- Unrealized P&L % and $ — direction-corrected (shorts profit when price falls)
- Entry date

Position quantities and prices come directly from the Binance Futures API via `fetch_binance_live_balances()`. Cached for 30 seconds. Falls back to cached OHLCV + state file if Binance is unreachable.

**Position Risk Tracker** — entry price, peak/trough reference, current price, signed P&L%, drawdown from reference, stop-loss and take-profit levels (correct direction for both longs and shorts).

**Signal Heatmap** — a strategies × tokens matrix showing each strategy's current signal strength. Green = long, Red = short, Grey = neutral. Active strategies marked with ★.

**Live Trading History** — actual orders placed on Binance Futures Demo, color-coded (green BUY, red SELL), with realized P&L for closing trades.

**Hypothetical Strategy Competition** — all 10 strategies running as paper portfolios. Ranked by 7-day return. Each shows its own NAV curve, current weights, trade history, and realized P&L. This is how the live feedback loop is visualized.

**Auto-refresh:** Dashboard reads state every 30 seconds. Binance balance data also cached 30 seconds.

### Strategy Monitor
Signal snapshots, regime analysis, and seasonality data for all 10 strategies.

### JSON corruption recovery
If `live_state.json` is corrupted (rare — only if the server crashes mid-write), the dashboard uses `json.JSONDecoder().raw_decode()` to recover the first valid JSON object from the file. Atomic writes (`.tmp` → rename) prevent this in normal operation.

---

## 17. Performance Attribution

**File:** `attribution/factor_model.py`

For each strategy's return series, runs OLS regression against 5 constructed factors:

| Factor | Construction |
|--------|-------------|
| `btc_ret` | BTC daily log return |
| `eth_ret` | ETH daily log return |
| `vol_factor` | 10-day realized vol change |
| `mom_factor` | Equal-weight top-5 minus bottom-5 returns |
| `carry` | Average daily funding rate across universe |

**Output per strategy:**
- Factor betas (how much each factor explains the return)
- Alpha (annualized unexplained return — what the strategy adds beyond factors)
- R² (how much variance is explained by the 5 factors)
- Information Ratio (alpha / tracking error)

**Hedging analysis:** What happens to Sharpe if we hedge out BTC exposure (set BTC beta to 0)? This shows how much of each strategy's performance comes from just being long crypto vs. actual alpha.

---

## 18. Data Layer

**File:** `data/ingestion.py`

### OHLCV data
- Source: Binance Klines API (`/api/v3/klines`)
- Interval: `1d` (daily bars) for backtest
- Coverage: `BACKTEST_START = "2023-01-01"` → today
- Cache: Parquet files in `data/cache/` with 6-hour TTL

```python
get_universe_ohlcv(use_cache=True)    # returns dict[symbol → DataFrame]
build_close_matrix(universe_data)      # returns DataFrame (dates × symbols)
build_return_matrix(close)             # log returns
```

### Funding rates
- Source: Binance USDT-margined perpetuals (`/fapi/v1/fundingRate`)
- 8-hour funding rates, reshaped to daily averages
- Used by Carry strategy

### Fear & Greed Index
- Source: `https://api.alternative.me/fng/`
- Returns integer 0–100 (0 = extreme fear, 100 = extreme greed)
- Free API, no authentication needed
- Falls back to neutral (50) if API is unavailable

### Token universe
12 tokens selected for liquidity, diversity, and coverage of major crypto categories:

| Token | Category | Rationale |
|-------|----------|-----------|
| BTCUSDT | L1 | Dominant macro signal |
| ETHUSDT | L1 | Smart contract platform |
| BNBUSDT | Exchange token | Binance ecosystem |
| SOLUSDT | L1 | High-throughput competitor to ETH |
| XRPUSDT | Payments | Institutional & cross-border |
| ADAUSDT | L1 | Academic/formal verification PoS |
| AVAXUSDT | L1/DeFi | Subnet architecture |
| DOTUSDT | Interoperability | Cross-chain IBC |
| LINKUSDT | Oracle | Data feeds infrastructure |
| POLUSDT | L2 | Polygon (replaced MATICUSDT Sep 2024) |
| ATOMUSDT | IBC ecosystem | Cross-chain hub |
| LTCUSDT | BTC proxy | Payments / store of value |

**Minimum volume filter:** Tokens with < $10M 24h volume are excluded to avoid illiquidity.

---

## 19. Configuration Reference

All parameters are in `config/settings.py`. Never hardcode values in strategy files.

### Data parameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `KLINE_INTERVAL` | `"1d"` | Daily bars for backtest |
| `BACKTEST_START` | `"2023-01-01"` | Backtest start (post-LUNA/FTX) |
| `BACKTEST_TEST_START` | `"2025-01-01"` | Out-of-sample period start |
| `BACKTEST_END` | `None` | None = today |
| `CACHE_EXPIRY_HOURS` | `6` | Re-fetch if cache older than 6h |
| `MIN_DAILY_VOLUME_USDT` | `10_000_000` | Liquidity filter |

### Risk parameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `MIN_ANNUALIZED_VOL` | `0.03` | 3% portfolio vol floor |
| `MAX_WEIGHT_SUM` | `1.00` | No leverage > 100% gross |
| `LONG_SHORT` | `True` | Allow short signals in backtest/hypotheticals |
| `RISK_LOOKBACK_DAYS` | `180` | 6-month covariance window |
| `MAX_POSITION_SIZE` | `0.20` | 20% single-token cap |
| `TRANSACTION_COST_BP` | `10` | 10 bps round-trip cost |

### Position risk
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `STOP_LOSS_PCT` | `0.02` | 2% hard stop (direction-aware) |
| `TAKE_PROFIT_PCT` | `0.03` | 3% take profit (direction-aware) |
| `TRAILING_STOP_PCT` | `0.05` | 5% trail from peak/trough |
| `USE_TRAILING_STOP` | `True` | Trailing stop enabled |

### Live engine
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `PAPER_TRADING` | `True` | Demo Futures only (never live) |
| `FUTURES_LEVERAGE` | `1` | Leverage set on all symbols at startup |
| `ORDER_TYPE` | `"MARKET"` | Market orders |
| `SLIPPAGE_BP` | `5` | 5 bps assumed slippage (backtest only) |
| `PORTFOLIO_USDT` | `10_000` | Hypothetical paper portfolio baseline only — live account capital read from Binance |
| `MIN_ORDER_USDT` | `11` | Minimum order value |
| `SIGNAL_RECOMPUTE_MINS` | `1` | Signal recompute frequency |
| `PRICE_MONITOR_SECS` | `60` | Stop/TP check frequency |
| `REBALANCE_THRESHOLD` | `0.03` | Minimum Σ\|Δweight\| to trade |
| `MAX_LIVE_POSITIONS` | `6` | Max simultaneous positions per side |

---

## 20. Output Files

All outputs written to `results/` after a backtest run.

| File | Format | Content |
|------|--------|---------|
| `strategy_metrics.json` | JSON | Nested: full / in_sample / out_of_sample metrics per strategy |
| `portfolio_returns.csv` | CSV | Daily return series, one column per strategy |
| `cumulative_returns.csv` | CSV | Cumulative NAV, one column per strategy |
| `monthly_seasonality.csv` | CSV | Average monthly Sharpe per strategy (index=month 1-12) |
| `regime_performance.csv` | CSV | Sharpe per strategy per regime (bull/bear/sideways) |
| `attribution_report.csv` | CSV | OLS betas, alpha, R², IR per strategy |
| `seasonality_report.csv` | CSV | Combined regime + monthly report |
| `nav_history_export.csv` | CSV | Live NAV history export (written on shutdown) |
| `pnl_summary.csv` | CSV | P&L breakdown per strategy |
| `live_state.json` | JSON | Live positions (signed), wallet balance, trades, NAV history, hypotheticals |

---

## 21. Deployment (Server)

### Initial deployment

```bash
# On server
cd /opt/crypto_algo
git pull origin main
source venv/bin/activate

# Create API key file
cat > Binance.env << 'EOF'
BINANCE_API_KEY=your_real_api_key
BINANCE_API_SECRET=your_real_api_secret
BINANCE_DEMO_API_KEY=your_demo_futures_key
BINANCE_DEMO_API_SECRET=your_demo_futures_secret
EOF

# Verify futures connectivity before starting
python3 debug_futures.py

# Clear any stale results from previous runs
rm -f results/*.csv results/*.json

# Start live engine in tmux (keeps running after SSH disconnect)
tmux new-session -d -s crypto "source venv/bin/activate && python3 main.py --mode full --run-now 2>&1 | tee logs/main.log"

# Start dashboard in separate window
tmux new-window -t crypto "source venv/bin/activate && streamlit run dashboard.py --server.port 8501"
```

### Pulling updates

```bash
cd /opt/crypto_algo && git pull
# Restart the live engine to pick up code changes:
tmux send-keys -t crypto C-c   # stop old session
tmux send-keys -t crypto "python3 main.py --mode full --run-now" Enter
```

No need to delete `live_state.json` on a normal code update. The engine reads the state file and continues from where it left off.

### Resetting the live state (fresh start)

Only do this if the state is corrupted or you want a fresh baseline:

```bash
# Delete all results and restart
rm -f results/*.csv results/*.json
python3 main.py --mode full --run-now
```

On restart with no state file, the engine calls `_bootstrap_from_binance()` to read actual token positions and wallet balance from Binance Futures and reconstruct state. `initial_nav` is stamped from the real account equity on the first reconciliation cycle.

### Viewing live logs

```bash
# Follow live engine logs
tail -f logs/live_engine.log

# Or if using tmux
tmux attach -t crypto
```

---

## 22. Common Questions & Troubleshooting

### Q: Why is NAV showing zero or stopping the engine immediately?
**A:** The engine calls `os._exit(1)` if NAV is zero. This means the Binance Futures Demo wallet is unfunded or the futures API call failed. Run `python3 debug_futures.py` to check connectivity. If `futures_account` fails, verify the demo API key has "Enable Futures" ticked in Binance API settings.

### Q: The system hasn't traded in 3+ days — why is it just holding?
**A:** Two possible causes:
1. `REBALANCE_THRESHOLD = 3%` — if the blended signal hasn't changed by at least 3% total weight, no orders fire. This is intentional to avoid churn.
2. The delta comparison uses actual live positions (from Binance reconciliation) vs new target. If your positions already match the target, delta will be near zero.

### Q: How do I know which strategy is currently active?
**A:** Check the Active Strategy Banner at the top of the Live Trading tab in the dashboard. It shows strategy names and blend percentages. You can also check `results/live_state.json` → `active_strategies` and `active_strategy_weights`.

### Q: Why does Risk Parity sometimes show "no signals"?
**A:** If a token in the universe has stale/zero price data (historically MATICUSDT after the MATIC→POL migration), its rolling volatility is zero, making inverse-vol infinite. The code handles this by replacing `inf` with `nan` and using `skipna=True` so the stale token is excluded. MATICUSDT has been replaced with POLUSDT in the universe.

### Q: Why do I get a RuntimeWarning about sqrt of negative number?
**A:** This comes from the portfolio optimizer computing `√(wᵀΣw)` where floating-point rounding in near-singular covariance matrices can produce a tiny negative number. All sqrt calls are guarded with `max(0.0, ...)`. These warnings do not affect results.

### Q: If I update the code and restart, do I lose my position history?
**A:** No. `live_state.json` persists on disk. The engine reads it on startup and continues from the last known state. Only delete the state file if you want a true fresh start.

### Q: Do profits get reinvested?
**A:** Yes. All profits stay in the portfolio. As the futures wallet balance grows from realized gains, the next rebalance sizes positions based on the higher NAV, compounding returns automatically.

### Q: What happens if a token gets delisted?
**A:** The engine fetches LOT_SIZE from `futures_exchange_info()` before placing any order. If a symbol is missing from the response, the token is skipped with a warning log. The strategy signal for that token should also degrade gracefully since its price history will stop updating.

### Q: Can this trade with real money?
**A:** Only if you set `PAPER_TRADING = False` in `config/settings.py` and replace the demo client keys with live keys. This should never be done without fully understanding the code and the risks. All current development uses `PAPER_TRADING = True` which routes all orders to Binance Futures Demo (virtual money, real prices).

### Q: How accurate is the backtest?
**A:** The backtest is realistic by design:
- T+1 execution lag (signals computed day T, orders placed day T+1)
- 10 bps round-trip transaction costs
- 5 bps slippage assumption
- 180-day rolling covariance (not full-history, which would look-ahead in practice)
- ML strategy uses strict walk-forward retraining every 30 days with no target leakage
- In-sample / out-of-sample split at 2025-01-01 for honest evaluation

Known limitations: daily bars mean intraday price impact is not modeled; funding costs for short positions are not subtracted in the backtest P&L; corporate events (forks, airdrops) are ignored.

### Q: What does the `--run-now` flag do?
**A:** By default, the live engine waits until the next scheduled slot (every 1 minute). `--run-now` fires one rebalance immediately on startup before waiting for the scheduler. Useful for testing or after a long outage.

### Q: How do I check if the API keys are working?
**A:** Run `python3 debug_futures.py`. It checks both the real Binance connection (for market data) and the demo futures connection (for orders and account balance), and prints your wallet balance. Use this any time you suspect a connectivity issue.

### Q: The dashboard says "Binance unavailable" — what does that mean?
**A:** `fetch_binance_live_balances()` failed to reach the Binance Futures API. The dashboard automatically falls back to reading positions from the state file and prices from cached OHLCV parquet files. This fallback is clearly labeled in the UI. Check your demo API key and run `python3 debug_futures.py` to diagnose.

### Q: Why is cash sometimes slightly off from what I expect?
**A:** Cash (`cash_usdt`) is the USDT wallet balance from Binance Futures, reconciled every cycle. Minor discrepancies between cycles can arise from lot-size rounding and fee deduction timing. These differences are small and self-correct on the next reconciliation.

### Q: How do I add a new strategy?
1. Create a new class in `strategies/alpha.py` inheriting from `BaseStrategy`
2. Implement `generate_signals(self, close, returns, **kwargs) → pd.DataFrame`
3. Add parameters to `STRATEGY_PARAMS` in `config/settings.py`
4. Add the class to `get_all_strategies()` at the bottom of `alpha.py`
5. The strategy automatically gets added to the backtest, hypothetical competition, and dashboard

### Q: How do I change the leverage?
**A:** Change `FUTURES_LEVERAGE = 1` in `config/settings.py`. The engine calls `futures_change_leverage()` for all 12 symbols at startup. Keep in mind that leverage above 1x amplifies both gains and losses and tightens effective stop-loss distances.
