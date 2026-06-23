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
7. [The 15 Alpha Strategies](#7-the-15-alpha-strategies)
8. [How Signals Become Orders](#8-how-signals-become-orders)
9. [Backtest Engine](#9-backtest-engine)
10. [Portfolio Optimizer](#10-portfolio-optimizer)
11. [Research Gate & Strategy Evaluation](#11-research-gate--strategy-evaluation)
12. [Hypothetical Paper Portfolios](#12-hypothetical-paper-portfolios)
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

**15 alpha strategies** are implemented and evaluated through a strict out-of-sample research gate. Exactly **one strategy passed** — 12-1 momentum with a banded vol-targeting overlay (`momentum_vt`) — and it alone is deployed in live trading. The live engine trades only that strategy; the Max-Sharpe optimizer is backtest-only and never runs in the live path.

All 15 strategies continue to run as hypothetical paper portfolios for ongoing research evaluation, but their live performance does **not** influence which strategy is active — the live book is fixed to whatever has cleared the OOS gate.

```
Market Data → 15 Strategies → Backtest → OOS Gate → 1 strategy passes → Live Engine → Orders
                   │
                   └── Hypothetical paper portfolios (research evaluation only, not live selection)
```

**Key design decisions:**
- **OOS gate before live** — a strategy must clear gross OOS Sharpe > 0.5, net OOS Sharpe > 0, and turnover < 1000% before it can enter `LIVE_BOOK_STRATEGIES`; currently only `momentum_vt` passes
- **Futures Demo Trading** — `PAPER_TRADING = True` is hardcoded as the default; virtual funds on `testnet.binancefuture.com`, real market prices
- **Long and short** — futures perpetuals allow real short positions; live engine uses `long_short=True`
- **Signal-driven, not clock-driven** — orders only fire when signal weight delta exceeds `REBALANCE_THRESHOLD`
- **VT overlay in live path** — bare momentum weights are scaled by a banded vol-target overlay (target 15%, ±20% band) matching the backtest overlay exactly; scale persisted in PostgreSQL across restarts
- **Optimizer is backtest-only** — the live weight path uses proportional sizing + `MAX_POSITION_SIZE` cap, never the Max-Sharpe SLSQP optimizer; `MIN_ANNUALIZED_VOL` is inert live
- **Zero look-ahead** — signals are shifted 1 day before weights are computed; ML model trained on strictly past data
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
        │  State: PostgreSQL live_state table (via db/state.py)   │
        └───────────────────────────────────────────────────┘
                │
                ▼
        ┌───────────────────────────────────────────────────┐
        │           DASHBOARD (dashboard/app.py)            │
        │                                                   │
        │  python3 dashboard/app.py  (Dash, port 8050)      │
        │  ├── Backtest Analysis tab                        │
        │  ├── Live Trading (Futures) tab                   │
        │  │     Reads PostgreSQL live_state table          │
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
├── Binance.env                    ← API keys (gitignored)
│
├── dashboard/
│   ├── app.py                     ← Dash dashboard (port 8050)
│   └── data.py                    ← Dashboard data-access helpers (DB + JSON fallback)
│
├── diagnostics/
│   ├── debug_futures.py           ← Futures API connectivity diagnostic
│   └── Binance test.py            ← Manual API test script
│
├── config/
│   ├── settings.py                ← ALL tuneable parameters
│   ├── client.py                  ← Dual client: real data / demo futures orders
│   └── requirements.txt           ← Python dependencies
│
├── data/
│   ├── ingestion.py               ← OHLCV, funding rates, Fear & Greed fetch
│   └── cache/                     ← Auto-generated Parquet files (6h TTL)
│
├── strategies/
│   ├── base.py                    ← BaseStrategy abstract class
│   └── alpha.py                   ← All 15 strategies
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
├── tests/
│   └── test_vt_equivalence.py     ← Equivalence tests for VT overlay live port
│
├── results/                       ← Auto-generated (gitignored)
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
pip install -r config/requirements.txt
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
pip install -r config/requirements.txt

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

# Launch the interactive dashboard (separate terminal, port 8050)
python3 dashboard/app.py
```

### What `--mode full` does step by step:
1. Fetches OHLCV data for all 12 tokens (from cache if fresh)
2. Fetches funding rates and Fear & Greed index
3. Generates signals for all 15 strategies
4. Runs walk-forward backtest for each strategy (IS and OOS split)
5. Applies VT overlay to momentum and reports `momentum_vt` gate result
6. Computes regime + monthly seasonality scores (backtest diagnostics)
7. Runs factor attribution
8. Saves all results to `results/`
9. Sets leverage for all universe symbols via Binance Futures API
10. Starts the APScheduler live engine (blocking — keeps running until Ctrl+C)

---

## 7. The 15 Alpha Strategies

All strategies live in `strategies/alpha.py` and inherit from `BaseStrategy`. Each outputs a signal DataFrame of shape `(dates × tokens)` with values in `[-1, +1]`.

**Live status:** Only `momentum_vt` (Strategy 1 with the VT overlay, Strategy 15 below) is in `LIVE_BOOK_STRATEGIES`. The remaining 14 run as hypothetical paper portfolios for ongoing research evaluation. A strategy can be promoted to live only by clearing the OOS gate (see Section 11).

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

### Strategy 3 — Cross-Sectional Momentum (`cross_sectional_momentum`)
**What it does:** Ranks all 12 tokens by their 20-day return each day. The rank is rescaled to `[-1, +1]` continuously — the top-ranked token gets `+1`, the bottom gets `-1`.

**Why it works:** Relative momentum within the same asset class removes the systematic market beta (you're long relative winners and short relative losers). Unlike time-series momentum, this strategy is always market-neutral in signal space.

**Key parameters:**
- `lookback = 20` — 20-day return ranking window
- `rank_method = "min"` — ties broken by minimum rank

---

### Strategy 4 — Volatility Breakout (`vol_breakout`) ⚠️ DISABLED
**What it does:** ATR (Average True Range) channel breakout. If today's close is above yesterday's close + (ATR × multiplier), the signal is positive (long breakout). If below yesterday's close - (ATR × multiplier), it is negative (short breakdown).

**Why it works:** Volatility compression followed by expansion signals the start of a new directional trend. Common in technical trading; effective in crypto where vol clustering is pronounced.

**Key parameters:**
- `atr_period = 14` — rolling window for ATR calculation
- `atr_multiplier = 2.0` — ATR band half-width multiplier
- `lookback = 20`

**Signal formula:** `(close - prev_close) / (atr_mult × ATR)` clipped to `[-1, +1]`. Continuously graded — a large breakout gets a signal near ±1.

**Status:** Excluded from `get_all_strategies()` — OOS win-rate 30.5%, below the 40% minimum threshold. Coded but not run.

---

### Strategy 5 — Pairs Trading (`pairs_trading`)
**What it does:** BTC/ETH cointegration spread trade. Estimates the hedge ratio β via OLS regression on a rolling 60-day window: `spread = log(BTC) - β × log(ETH)`. When the spread is wide (high z-score), it shorts BTC and longs ETH. When spread is narrow (low z-score), the opposite.

**Why it works:** BTC and ETH share systematic crypto risk factors (macro, regulation, sentiment). Their idiosyncratic spread tends to mean-revert. The rolling OLS hedge ratio adapts to structural changes in the relationship.

**Key parameters:**
- `pair = ("BTCUSDT", "ETHUSDT")`
- `lookback = 60` — rolling OLS window
- `entry_z = 2.0` — z-score of ±2 maps to signal ±1
- `exit_z = 0.0` — signal returns to 0 when spread returns to mean

**Note:** The cointegration p-value gate (`coint_pvalue = 0.05`) in config is currently not enforced — the signal is always live. This was intentional to avoid the strategy going silent during regime breaks.

---

### Strategy 6 — ML Signal (`ml_signal`)
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

### Strategy 7 — Macro Rotation (`macro_rotation`)
**What it does:** Uses BTC's 20-day return as a macro risk-on/risk-off indicator. If BTC's recent return is positive (risk-on), all tokens get positive signals proportional to their own recent return. If BTC is negative (risk-off), signals are suppressed or reversed.

**Why it works:** BTC is the dominant systematic factor in crypto. Its trend strongly influences altcoin performance. This strategy is effectively a market-regime filter that scales exposure based on macro conditions.

**Key parameters:**
- `risk_on_threshold = 0.0` — positive 20d BTC return = risk-on
- `lookback = 20` — BTC return lookback
- `btc_proxy = "BTCUSDT"`

---

### Strategy 8 — Carry (`carry`)
**What it does:** Uses Binance perpetual futures 8-hour funding rates as a carry signal. Tokens with consistently positive funding rates (longs paying shorts) are in demand — this is bullish. Tokens with negative funding rates are under short pressure — bearish.

**Why it works:** Perpetual funding rates are the crypto equivalent of the carry factor in FX. Positive funding = market is net long, reflecting bullish sentiment. Negative funding = market is net short.

**Key parameters:**
- `lookback = 7` — 7-day average funding rate
- `top_n = 4` — long top 4 by funding carry

**Data source:** `data/ingestion.py → get_universe_funding_rates()` which calls Binance `/fapi/v1/fundingRate`. Falls back gracefully if the endpoint is unavailable.

---

### Strategy 9 — Sentiment (`sentiment`)
**What it does:** Uses the Alternative.me Crypto Fear & Greed Index (0–100). When the index is above 60 (greed), the market is risk-on — all tokens get a positive signal. When below 30 (fear), it uses a contrarian long signal. Between 30–60 = neutral.

**Why it works:** The Fear & Greed Index captures collective market sentiment. Extreme fear often precedes reversals (contrarian); greed during established trends can reinforce momentum.

**Key parameters:**
- `greed_threshold = 60` — above this → risk-on long bias
- `fear_threshold = 30` — below this → contrarian long
- `lookback = 7` — trailing average of the index

**Data source:** `https://api.alternative.me/fng/` — free, no authentication required.

---

### Strategy 10 — Exhaustion Fade (`exhaustion_fade`)
**What it does:** Waits for a token to overextend — confirmed by a Bollinger Band breach-then-close-back-inside pattern, extreme perpetual funding (overcrowded positioning), and a low ADX (ranging market). All three conditions must align on the same bar before a signal is issued.

**Signal direction:**
- `+1` → price closed back inside from below the lower band + negative funding (shorts overcrowded) → long the snapback
- `-1` → price closed back inside from above the upper band + positive funding (longs overcrowded) → short the snapback

**Why it works:** Three-condition confirmation filters out low-quality setups. The funding gate ensures the crowd is on the wrong side; the ADX gate ensures there is no strong trend to fight against.

**Key parameters:**
- `bb_window = 20`, `bb_std = 2.0` — Bollinger Band width
- `adx_period = 14`, `adx_threshold = 25` — ADX ranging filter
- `funding_threshold` — per-8hr funding extremity gate

**Signal strength:** Scaled by ADX distance from threshold and funding extremity — cleaner setups receive larger allocations (0.3–1.0 range).

---

### Research Candidates (Strategies 11–14)

The following four strategies are coded and run as paper portfolios but have not cleared the OOS research gate. They are tagged `# RESEARCH CANDIDATES` in `alpha.py`.

#### R1 — Vol-Scaled TSMOM (`tsmom_volscaled`)
Sign of the 12-month trailing return divided by each token's own 3-month realized vol. Low-vol tokens receive larger allocations. Differs from Strategy 1 in that sizing is continuous and vol-driven rather than binary rank-based. Cross-sectionally normalized to `[-1, +1]` per bar.

#### R2 — Carry Neutral (`carry_neutral`)
Dollar-neutral variant of Strategy 8. Cross-sectionally demeans funding rates (each token's funding minus the universe mean) before computing the signal. Net exposure sums to zero across the universe by construction — only relative carry is traded, not directional funding beta.

#### R3 — Residual Momentum (`resid_momentum`)
Rolling OLS strips BTC's systematic return factor from each token's daily return. Momentum signal is then computed on the BTC-factor-neutralized residuals. Captures token-specific winner/loser dynamics without the dominant BTC beta contaminating the signal.

#### R4 — BTC Dominance (`btc_dominance`)
Tracks BTC's share of total universe price-weighted market action. Rising BTC dominance → capital rotating from altcoins into BTC (risk-off): long BTC, short altcoins. Falling dominance → speculative appetite expanding: short BTC, long altcoins. Signal magnitude is the z-score of the smoothed daily dominance change.

#### R5 — Vol Spike Reversion (`vol_spike_reversion`)
Fades extreme single-day moves (|return| > `spike_mult × ATR`) when perpetual funding confirms the crowd is positioned in the same direction as the spike. Signal is held for `hold_bars` days (time stop). Intentionally a left-tail-focused strategy: low average P&L per bar but the entries are high-volatility by construction.

---

### Strategy 15 — Momentum VT (`momentum_vt`) — Live
**What it is:** A synthetic overlay result, not a standalone strategy class. Created by the backtest pipeline (`overlay_backtest_result`) by applying the vol-target overlay (`apply_vol_target_overlay`) to the raw `momentum` signals.

**Why it exists separately:** The backtest tracks both the un-overlaid `momentum` equity curve and the VT-overlaid `momentum_vt` curve. The OOS gate is evaluated on `momentum_vt` (not raw momentum) because the live engine applies the VT scale. This ensures backtest and live metrics are directly comparable.

**Live implementation:** `compute_live_vt_scale` in `portfolio/optimizer.py` — mathematically equivalent to the batch overlay, starting from a Postgres-persisted band state. Proven equivalent in `tests/test_vt_equivalence.py`.

---

## 8. How Signals Become Orders

This is the full live pipeline from momentum signal to a Binance Futures order.

### Step 1 — Signal generation
`momentum` strategy outputs a float in `[-1, +1]` per token from its 12-1 month lookback. Binary ±1 (top-4 long, bottom-4 short, others zero).

### Step 2 — Weight conversion (`_signal_to_weights`)
Signals are converted to portfolio weights with `long_short=True` (futures supports real shorts):
- Top `MAX_LIVE_POSITIONS=6` positive signals → long positions (weights > 0)
- Top `MAX_LIVE_POSITIONS=6` negative signals → short positions (weights < 0)
- Long and short each get up to 50% of gross exposure when both sides are present
- Any token exceeding `MAX_POSITION_SIZE = 20%` is capped iteratively; excess redistributed

Example:
```
Signals:  BTC=1.0, SOL=1.0, ETH=-1.0, ADA=-1.0
Long budget: 50% → BTC: 25%, SOL: 25%
Short budget: 50% → ETH: -25%, ADA: -25%
```

### Step 3 — Vol-target overlay (`compute_live_vt_scale`)
The bare weights are scaled by a single scalar so realized 63-day portfolio vol stays near 15% (±20% band). The scalar is capped at 2× and persisted in PostgreSQL across cycles. This step is mathematically equivalent to the backtest's `overlay_backtest_result()` — proven in `tests/test_vt_equivalence.py`.

### Step 4 — Price fetch (live futures ticker)
The optimizer is **not** called. `MIN_ANNUALIZED_VOL` is inert in the live path.
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

**Live path:** The optimizer is **backtest-only**. `compute_target_weights()` in the live engine uses proportional sizing + `MAX_POSITION_SIZE` cap, not SLSQP. `MIN_ANNUALIZED_VOL = 0.03` is enforced only inside the optimizer and is therefore inert during live trading.

---

## 11. Research Gate & Strategy Evaluation

**File:** `seasonality/analyzer.py`, `backtest/engine.py`

All 15 strategies are evaluated through a strict three-criterion out-of-sample gate before any strategy may enter `LIVE_BOOK_STRATEGIES`. A strategy must clear **all three** simultaneously:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Gross OOS Sharpe | > 0.5 | Signal must be economically meaningful before costs |
| Net OOS Sharpe | > 0.0 | Must be profitable after 10 bp round-trip + 15 bp slippage |
| Annual turnover | < 1000% | Prevents high-churn strategies whose costs dominate |

The OOS period is `BACKTEST_TEST_START = 2025-01-01` → today — data the strategies never saw during development.

**Gate result (as of current backtest):** One strategy passed — `momentum_vt` (12-1 momentum with VT overlay). All others failed on at least one criterion, most commonly net OOS Sharpe ≤ 0.

The gate result is printed by `main.py` at the end of each backtest run and shown in the dashboard's Backtest Analysis tab OOS gate table. Promotion to live is manual — clearing the gate flags a strategy as eligible but does not automatically add it to `LIVE_BOOK_STRATEGIES`.

### Regime and seasonality diagnostics (backtest reporting only)
`seasonality/analyzer.py` computes per-strategy regime-conditional Sharpe (bull/bear/sideways based on BTC 200-day MA + realized vol percentile) and monthly seasonality scores. These are **backtest diagnostics** — they appear in the dashboard's regime and seasonality heatmaps and inform when a strategy's edge is likely to be present, but they do not influence which strategy is active in the live engine.

---

## 12. Hypothetical Paper Portfolios

Every `signal_rebalance_job` cycle, all 15 strategies run as independent hypothetical paper portfolios. These are **research instruments**, not live-selection inputs.

### Purpose
- Provide forward-looking OOS performance data to re-evaluate the gate periodically
- Detect if a non-live strategy has started consistently outperforming (human review trigger)
- Feed the dashboard's Strategy Competition panel for visibility

### NAV update (each cycle)
```python
period_return = Σ prev_weight[sym] × (price_now - price_prev) / price_prev
new_nav       = prev_nav × (1 + period_return)
```
Shorts are modeled correctly: negative weights produce negative returns when prices rise.

### Storage
Persisted in `hypothetical_nav` and `hypothetical_strategy_state` Postgres tables. NAV history loaded up to 2880 rows per strategy; trade history up to 500 rows. Weights and prices are checkpointed so NAV continuity survives engine restarts.

### What they do NOT do
Paper portfolio results do **not** change which strategy is active. `LIVE_BOOK_STRATEGIES` is a static config value updated only after manual review of the research gate.

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
2. Recompute signals for all 15 strategies
3. Snapshot latest signals to state (dashboard reads these)
4. Fetch current futures prices
5. **Reconcile with Binance Futures** — sync positions (signed qty), wallet balance, stamp `initial_nav` on first cycle
6. Update all 15 hypothetical paper portfolios (research evaluation, not live selection)
7. Compute new target weights for live portfolio (`LIVE_BOOK_STRATEGIES = ["momentum"]`): signal → proportional weights → VT overlay → clip
8. Compare new weights to actual current positions
9. If `Σ|Δweight| > REBALANCE_THRESHOLD (15%)` → execute futures orders; else skip

### Why signal-driven (not clock-driven)?
If BTC surges 10% at 3am, the system rebalances within 1 minute regardless of schedule. If the market is flat, no unnecessary trading occurs even when the scheduler fires.

### State persistence — PostgreSQL
State is stored in PostgreSQL (configured via `DB_URL` in `config/settings.py`, managed via `db/state.py`). Key tables:

| Table | Content |
|-------|---------|
| `live_state` | Core row: positions (signed), cash, weights, active strategies, VT scale |
| `position_entries` | Per-symbol entry price, date, peak price |
| `nav_history` | Timestamped NAV series (last 2880 rows loaded) |
| `trade_log` | All executed orders with P&L |
| `hypothetical_nav` | Per-strategy paper portfolio NAV history |
| `hypothetical_trades` | Per-strategy paper portfolio trade history |
| `hypothetical_strategy_state` | Per-strategy weights/prices for NAV continuity across restarts |

`active_strategy_weights` also stores `_vt_scale` — the persisted vol-target scaling factor (see VT overlay below).

Note: `positions[sym]` is **signed** — positive = long, negative = short. `cash_usdt` = futures wallet balance (USDT), synced from Binance each cycle.

### Vol-target overlay in the live path
After computing bare momentum weights, `compute_target_weights()` applies the same vol-targeting overlay as the backtest's `overlay_backtest_result()` (parameters from `VOL_TARGET_PARAMS`):

```
target_vol = 15%,  band = ±20%  (corridor: 12%–18%)
vol_window = 63 days,  max_scale = 2×
```

The overlay scales the entire weight vector so realized 63-day portfolio vol stays near 15%. The scalar only updates when realized vol exits the ±20% band (reduces turnover). The current scale is persisted in `active_strategy_weights["_vt_scale"]` in PostgreSQL so the band state survives engine restarts without recomputing the full history.

Equivalence is proven in `tests/test_vt_equivalence.py` (4 tests, rtol=1e-12): given the same weights and returns, the live incremental path produces scales identical to the batch backtest overlay.

The live weight path **never** calls the max-Sharpe optimizer — `MIN_ANNUALIZED_VOL` remains inert in the live path.

### Bootstrapping on fresh start (no Postgres row)
When the `live_state` Postgres table has no row yet, `load_state()` calls `_bootstrap_from_binance()`:
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
| `STOP_LOSS_PCT = 0.06` | 6% | Exit if signed P&L drops below -6% |
| `TAKE_PROFIT_PCT = 0.12` | 12% | Exit if signed P&L exceeds +12% |
| `TRAILING_STOP_PCT = 0.10` | 10% | Exit if price moves 10% against peak/trough |
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

**File:** `dashboard/app.py`  
**Command:** `python3 dashboard/app.py`  
**URL:** `http://<server-ip>:8050`

Built with Plotly Dash. Data is sourced from PostgreSQL via `dashboard/data.py`, with automatic fallback to `results/strategy_metrics.json` and CSV files when the DB has no rows (e.g., immediately after a fresh backtest before the DB is populated).

### Backtest Analysis tab
- OOS research gate table — strategies ranked by gross/net OOS Sharpe; PASS/FAIL against `RESEARCH_GATE` thresholds
- Strategy performance comparison (Sharpe, CAGR, Max Drawdown, Sortino, Calmar) — full period, IS, OOS
- Cumulative NAV curves for all strategies
- In-sample vs Out-of-sample comparison table
- Monthly seasonality heatmap (strategy × month)
- Regime performance heatmap (strategy × bull/bear/sideways)
- Factor attribution table (beta to BTC, ETH, vol, momentum, carry)

### Live Trading tab

**Active Strategy Banner** — which strategies are live-book active with blend percentages. Driven by `LIVE_BOOK_STRATEGIES` (currently `["momentum"]`).

**NAV Metrics** — headline numbers from the PostgreSQL `live_state` table:
- Futures NAV (wallet balance + unrealized PnL)
- Wallet Balance (USDT from Binance Futures — includes all realized PnL)
- Unrealized PnL, open positions count, total P&L vs `initial_nav`

**NAV History Chart** — time series from `nav_history` Postgres table.

**Open Positions Table** — token, side (LONG/SHORT), quantity, entry price, live price, market value, unrealized P&L % and $ (direction-corrected for shorts), entry date.

**Position Risk Tracker** — entry price, peak/trough reference, current price, signed P&L%, stop-loss and take-profit levels.

**Signal Heatmap** — strategies × tokens matrix showing current signal strength. Green = long, Red = short, Grey = neutral.

**Live Trading History** — actual Binance Futures Demo orders from the `trade_log` table.

**Hypothetical Strategy Competition** — all strategies running as paper portfolios tracked in `hypothetical_nav` Postgres table. Each shows NAV curve, current weights, and trade history.

### Strategy Monitor tab
Signal snapshots, regime analysis, and seasonality data for all strategies.

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
| `STOP_LOSS_PCT` | `0.06` | 6% hard stop (direction-aware) |
| `TAKE_PROFIT_PCT` | `0.12` | 12% take profit (direction-aware) |
| `TRAILING_STOP_PCT` | `0.10` | 10% trail from peak/trough |
| `USE_TRAILING_STOP` | `True` | Trailing stop enabled |

### Live engine
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `PAPER_TRADING` | `True` | Demo Futures only (never live) |
| `FUTURES_LEVERAGE` | `1` | Leverage set on all symbols at startup |
| `ORDER_TYPE` | `"MARKET"` | Market orders |
| `SLIPPAGE_BP` | `15` | 15 bps assumed slippage (realistic for thin alts) |
| `PORTFOLIO_USDT` | `10_000` | Hypothetical paper portfolio baseline only — live account capital read from Binance |
| `MIN_ORDER_USDT` | `11` | Minimum order value |
| `SIGNAL_RECOMPUTE_MINS` | `1` | Signal recompute frequency |
| `PRICE_MONITOR_SECS` | `300` | Stop/TP check frequency (every 5 min) |
| `REBALANCE_THRESHOLD` | `0.15` | Minimum Σ\|Δweight\| to trade (15%) |
| `MAX_LIVE_POSITIONS` | `6` | Max simultaneous positions per side |
| `LIVE_BOOK_STRATEGIES` | `["momentum"]` | Strategies active in live trading |
| `LIVE_REBALANCE_FREQ_DAYS` | `7` | Minimum days between live rebalances |

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

### Server setup (DigitalOcean / Ubuntu)

The system runs as three persistent `screen` sessions on `/opt/crypto_algo`.

```bash
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

# Verify connectivity before starting
python3 debug_futures.py

# Start live engine
screen -S engine
python3 main.py --mode live --run-now 2>&1 | tee -a logs/engine.log
# Ctrl+A, D to detach

# Start Dash dashboard (port 8050)
screen -S dashboard
python3 dashboard/app.py 2>&1 | tee -a logs/dashboard.log
# Ctrl+A, D to detach
```

Open port 8050 if not already open:
```bash
ufw allow 8050/tcp
```

Access at `http://<server-ip>:8050`.

### Pulling updates

```bash
cd /opt/crypto_algo && git pull

# Restart dashboard to pick up code changes:
screen -r dashboard   # Ctrl+C to stop
python3 dashboard/app.py 2>&1 | tee -a logs/dashboard.log
# Ctrl+A, D

# Restart engine only if engine code changed:
screen -r engine      # Ctrl+C to stop
python3 main.py --mode live --run-now 2>&1 | tee -a logs/engine.log
# Ctrl+A, D
```

Postgres state persists across restarts — the engine picks up from the last saved state automatically.

### Running a fresh backtest

Stop the engine first to avoid Binance rate-limit bans (both processes hammering the API simultaneously triggers IP bans):

```bash
screen -r engine      # Ctrl+C
python3 main.py --mode backtest 2>&1 | tee -a logs/backtest.log
# After it finishes, restart the engine:
python3 main.py --mode live --run-now 2>&1 | tee -a logs/engine.log
# Ctrl+A, D
```

### Resetting the live state (fresh start)

Only do this if you want a true fresh baseline:

```bash
# Truncate the live_state Postgres tables
psql -U sharonlau15 -d crypto_algo -c "
  DELETE FROM live_state;
  DELETE FROM position_entries;
  DELETE FROM nav_history;
  DELETE FROM trade_log;
"
# Also clear result files if desired
rm -f results/*.csv results/*.json
python3 main.py --mode full --run-now 2>&1 | tee -a logs/engine.log
```

On first run with no `live_state` row, the engine calls `_bootstrap_from_binance()` to read actual positions and wallet balance and stamp `initial_nav`.

### Viewing live logs

```bash
tail -f /opt/crypto_algo/logs/engine.log
tail -f /opt/crypto_algo/logs/dashboard.log
# Re-attach to a session:
screen -r engine
screen -r dashboard
```

---

## 22. Common Questions & Troubleshooting

### Q: Why is NAV showing zero or stopping the engine immediately?
**A:** The engine calls `os._exit(1)` if NAV is zero. This means the Binance Futures Demo wallet is unfunded or the futures API call failed. Run `python3 debug_futures.py` to check connectivity. If `futures_account` fails, verify the demo API key has "Enable Futures" ticked in Binance API settings.

### Q: The system hasn't traded in 3+ days — why is it just holding?
**A:** Two possible causes:
1. `REBALANCE_THRESHOLD = 15%` — momentum signals are binary ±1 and change only when a token crosses rank boundaries. If the signal hasn't changed by at least 15% total weight, no orders fire. This is intentional — momentum is a slow-moving signal.
2. The delta comparison uses actual live positions (from Binance reconciliation) vs new target. If your positions already match the target, delta will be near zero.

### Q: How do I know which strategy is currently active?
**A:** Check the Active Strategy Banner at the top of the Live Trading tab in the dashboard (`http://<server-ip>:8050`). It shows strategy names and blend percentages. You can also query Postgres: `SELECT active_strategies, active_strategy_weights FROM live_state WHERE id=1;`

### Q: Why does Risk Parity sometimes show "no signals"?
**A:** If a token in the universe has stale/zero price data (historically MATICUSDT after the MATIC→POL migration), its rolling volatility is zero, making inverse-vol infinite. The code handles this by replacing `inf` with `nan` and using `skipna=True` so the stale token is excluded. MATICUSDT has been replaced with POLUSDT in the universe.

### Q: Why do I get a RuntimeWarning about sqrt of negative number?
**A:** This comes from the portfolio optimizer computing `√(wᵀΣw)` where floating-point rounding in near-singular covariance matrices can produce a tiny negative number. All sqrt calls are guarded with `max(0.0, ...)`. These warnings do not affect results.

### Q: If I update the code and restart, do I lose my position history?
**A:** No. State is stored in PostgreSQL and persists across restarts. The engine reads from the `live_state` table on startup and continues from where it left off. Only truncate the DB tables if you want a true fresh start.

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
- 15 bps slippage assumption (realistic for thinner alts; total cost 25 bps)
- 180-day rolling covariance (not full-history, which would look-ahead in practice)
- ML strategy uses strict walk-forward retraining every 30 days with no target leakage
- In-sample / out-of-sample split at 2025-01-01 for honest evaluation

Known limitations: daily bars mean intraday price impact is not modeled; funding costs for short positions are not subtracted in the backtest P&L; corporate events (forks, airdrops) are ignored.

### Q: What does the `--run-now` flag do?
**A:** By default, the live engine waits until the next scheduled slot (every 1 minute). `--run-now` fires one rebalance immediately on startup before waiting for the scheduler. Useful for testing or after a long outage.

### Q: How do I check if the API keys are working?
**A:** Run `python3 debug_futures.py`. It checks both the real Binance connection (for market data) and the demo futures connection (for orders and account balance), and prints your wallet balance. Use this any time you suspect a connectivity issue.

### Q: The dashboard says "Binance unavailable" — what does that mean?
**A:** `fetch_binance_live_balances()` failed to reach the Binance Futures API. The dashboard automatically falls back to reading positions from PostgreSQL and prices from cached OHLCV parquet files. This fallback is clearly labeled in the UI. Check your demo API key and run `python3 debug_futures.py` to diagnose. If the IP is banned (Binance rate-limit), check the ban expiry: `python3 -c "import time; print(time.time())"` and compare to the ban timestamp in the error.

### Q: Why is cash sometimes slightly off from what I expect?
**A:** Cash (`cash_usdt`) is the USDT wallet balance from Binance Futures, reconciled every cycle. Minor discrepancies between cycles can arise from lot-size rounding and fee deduction timing. These differences are small and self-correct on the next reconciliation.

### Q: How do I add a new strategy?
1. Create a new class in `strategies/alpha.py` inheriting from `BaseStrategy`
2. Implement `generate_signals(self, close, returns, **kwargs) → pd.DataFrame`
3. Add parameters to `STRATEGY_PARAMS` in `config/settings.py`
4. Add the class to `get_all_strategies()` at the bottom of `alpha.py`
5. The strategy automatically enters the backtest and hypothetical paper portfolio competition
6. To promote to live: re-run backtest, confirm it clears the OOS gate (gross Sharpe > 0.5, net Sharpe > 0, turnover < 1000%), then manually add it to `LIVE_BOOK_STRATEGIES` in `config/settings.py`

### Q: How do I change the leverage?
**A:** Change `FUTURES_LEVERAGE = 1` in `config/settings.py`. The engine calls `futures_change_leverage()` for all 12 symbols at startup. Keep in mind that leverage above 1x amplifies both gains and losses and tightens effective stop-loss distances.
