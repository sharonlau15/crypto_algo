"""
config/settings.py
==================
Central configuration for the entire trading system.
All tuneable parameters live here — never hardcode in strategy files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data" / "cache"
LOG_DIR    = ROOT_DIR / "logs"
RESULT_DIR = ROOT_DIR / "results"

for _d in [DATA_DIR, LOG_DIR, RESULT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Environment ────────────────────────────────────────────────────────────────
# Looks for Binance.env or .env in the project root.
# On the server: create /opt/crypto_algo/Binance.env with your keys.
# Locally:       create crypto_algo/Binance.env (it is gitignored).
_env_loaded = False
for _env_path in [ROOT_DIR / "Binance.env", ROOT_DIR / ".env"]:
    if _env_path.exists():
        load_dotenv(_env_path)
        _env_loaded = True
        break

if not _env_loaded:
    import warnings
    warnings.warn(
        f"No Binance.env or .env found in {ROOT_DIR}. "
        "Create one with BINANCE_API_KEY / BINANCE_API_SECRET / "
        "BINANCE_DEMO_API_KEY / BINANCE_DEMO_API_SECRET.",
        stacklevel=2,
    )

API_KEY         = os.getenv("BINANCE_API_KEY")
API_SECRET      = os.getenv("BINANCE_API_SECRET")
DEMO_API_KEY    = os.getenv("BINANCE_DEMO_API_KEY", API_KEY)
DEMO_API_SECRET = os.getenv("BINANCE_DEMO_API_SECRET", API_SECRET)

# ── Universe ───────────────────────────────────────────────────────────────────
# 12 tokens — liquid, diverse, covers large/mid/alt cap crypto
UNIVERSE = [
    "BTCUSDT",   # Bitcoin        — dominant macro signal
    "ETHUSDT",   # Ethereum       — smart contract leader
    "BNBUSDT",   # BNB            — exchange token
    "SOLUSDT",   # Solana         — high-throughput L1
    "XRPUSDT",   # XRP            — payments / institutional
    "ADAUSDT",   # Cardano        — PoS L1
    "AVAXUSDT",  # Avalanche      — L1 / DeFi
    "DOTUSDT",   # Polkadot       — interoperability
    "LINKUSDT",  # Chainlink      — oracle / data feeds
    "POLUSDT",   # Polygon (POL)  — L2 scaling (MATIC delisted Sep 2024)
    "ATOMUSDT",  # Cosmos         — IBC ecosystem
    "LTCUSDT",   # Litecoin       — BTC proxy / payments
]

# Exclude these if they go illiquid (< $10M 24h volume)
MIN_DAILY_VOLUME_USDT = 10_000_000

# ── Data ───────────────────────────────────────────────────────────────────────
KLINE_INTERVAL      = "1d"          # Daily bars for portfolio rebalancing
KLINE_INTERVAL_1H   = "1h"          # 1h bars for intraday signals
BACKTEST_START      = "2023-01-01"  # Post-LUNA/FTX: cleaner market structure
BACKTEST_TEST_START = "2025-01-01"  # Out-of-sample test period begins here
                                    # In-sample:  BACKTEST_START → BACKTEST_TEST_START
                                    # Out-of-sample: BACKTEST_TEST_START → today
BACKTEST_END        = None          # None = today
CACHE_EXPIRY_HOURS  = 6             # Re-fetch if cached data is older than this

# ── Risk & Portfolio ───────────────────────────────────────────────────────────
MIN_ANNUALIZED_VOL  = 0.03          # 3% floor as per project spec
MAX_WEIGHT_SUM      = 1.00          # |w|_1 ≤ 100%
LONG_SHORT          = True          # Allow short positions
DOLLAR_NEUTRAL      = False         # Not required per spec
RISK_LOOKBACK_DAYS  = 180           # 6M covariance — recent conditions matter more in crypto
REBALANCE_FREQ      = "1d"          # Daily rebalance (T+1 execution)
MAX_POSITION_SIZE   = 0.20          # Single token cap at 20% (forces ≥5 positions)
TRANSACTION_COST_BP = 10            # 10 bps round-trip (Binance maker/taker)

# ── Position-Level Risk Management ─────────────────────────────────────────────
STOP_LOSS_PCT       = 0.02          # 2% hard stop loss per position
TAKE_PROFIT_PCT     = 0.03          # 3% take profit target
TRAILING_STOP_PCT   = 0.05          # 5% trailing stop (0 = disabled)
USE_TRAILING_STOP   = True          # Whether to use trailing stops

# ── Execution ──────────────────────────────────────────────────────────────────
PAPER_TRADING       = True          # True = demo trading, False = live (NEVER set False in dev)
ORDER_TYPE          = "MARKET"      # MARKET | LIMIT
SLIPPAGE_BP         = 5             # Assumed slippage in backtest
PORTFOLIO_USDT      = 10_000        # Baseline for hypothetical paper portfolios only.
                                    # Live account capital is read from Binance — not this value.
MIN_ORDER_USDT      = 11            # Binance min notional + buffer

# ── Real-time live trading ─────────────────────────────────────────────────────
SIGNAL_RECOMPUTE_MINS = 1           # How often to recompute signals and check for rebalance
PRICE_MONITOR_SECS    = 60          # How often to poll prices for stop/TP monitoring
REBALANCE_THRESHOLD   = 0.03        # Min total weight change (sum of |Δw|) to trigger rebalance
MAX_LIVE_POSITIONS    = 6           # Max number of simultaneous long positions in live trading

# ── Strategy Params ────────────────────────────────────────────────────────────
STRATEGY_PARAMS = {
    "momentum": {
        "lookback_long":  252,   # 12-month return (annualised)
        "lookback_short": 21,    # Skip last month (reversal filter)
        "top_n":          4,     # Long top N tokens
        "bottom_n":       4,     # Short bottom N tokens
    },
    "mean_reversion": {
        "zscore_window":  20,
        "entry_z":        2.0,
        "exit_z":         0.5,
    },
    "risk_parity": {
        "lookback":       63,    # 3-month vol for weights
        "max_iter":       100,
    },
    "cross_sectional_momentum": {
        "lookback":       20,
        "rank_method":    "min",
    },
    "vol_breakout": {
        "atr_period":     14,
        "atr_multiplier": 2.0,
        "lookback":       20,
    },
    "pairs_trading": {
        "pair":           ("BTCUSDT", "ETHUSDT"),
        "lookback":       60,
        "entry_z":        2.0,
        "exit_z":         0.0,
        "coint_pvalue":   0.05,
    },
    "ml_signal": {
        "feature_lookbacks": [1, 3, 5, 10, 21],
        "train_window":      180,   # 6M rolling train — matches RISK_LOOKBACK_DAYS
        "n_estimators":      200,
        "max_depth":         4,
        "learning_rate":     0.05,
    },
    "macro_rotation": {
        "risk_on_threshold":  0.0,   # Positive 20d return = risk-on
        "lookback":           20,
        "btc_proxy":          "BTCUSDT",
    },
    "carry": {
        # Funding rate proxy: use 8h perpetual funding rate from Binance
        "lookback":           7,     # 7-day avg funding
        "top_n":              4,
    },
    "sentiment": {
        # Fear & Greed Index (via alternative.me API — free, no auth)
        "greed_threshold":    60,    # > 60 = risk-on
        "fear_threshold":     30,    # < 30 = contrarian long
        "lookback":           7,
    },
}

# ── Seasonality ────────────────────────────────────────────────────────────────
SEASONALITY_MIN_PERIODS  = 24   # Need at least 24 observations per season bucket
REGIME_MA_WINDOW         = 200  # BTC 200-day MA for bull/bear regime
STRATEGY_SELECT_METRIC   = "sharpe"  # sharpe | sortino | calmar

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL   = "INFO"
LOG_ROTATION = "1 week"
