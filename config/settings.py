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

DB_URL          = os.getenv("DB_URL", "postgresql://sharonlau15:sharon@localhost:5432/crypto_algo")

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
TRANSACTION_COST_BP = 10            # 10 bps round-trip (Binance taker 5bps × 2)

# ── Position-Level Risk Management ─────────────────────────────────────────────
STOP_LOSS_PCT       = 0.06          # 6% hard stop — daily-bar signals need room; 2% fired on noise
TAKE_PROFIT_PCT     = 0.12          # 12% TP — wide enough to let winners run on daily signals
TRAILING_STOP_PCT   = 0.10          # 10% trailing stop (0 = disabled)
USE_TRAILING_STOP   = True          # Whether to use trailing stops

# ── Execution ──────────────────────────────────────────────────────────────────
PAPER_TRADING       = True          # True = demo trading, False = live (NEVER set False in dev)
ORDER_TYPE          = "MARKET"      # MARKET | LIMIT
SLIPPAGE_BP         = 15            # Market-order slippage: 5bp was too optimistic for thin alts
                                    # (ATOM/DOT/ADA/LTC realistic: 10-20bp). Total cost = 25bp.
FUTURES_LEVERAGE    = 1             # 1x = spot-equivalent risk; increase only if intentional
PORTFOLIO_USDT      = 10_000        # Baseline for hypothetical paper portfolios only.
                                    # Live account capital is read from Binance — not this value.
MIN_ORDER_USDT      = 11            # Binance min notional + buffer

# ── Real-time live trading ─────────────────────────────────────────────────────
SIGNAL_RECOMPUTE_MINS = 1           # How often to recompute signals and check for rebalance
PRICE_MONITOR_SECS    = 300         # Poll every 5 min — daily-bar signals; 60s manufactured turnover
REBALANCE_THRESHOLD      = 0.15     # Min total weight change (sum of |Δw|) to trigger rebalance
MAX_LIVE_POSITIONS       = 6        # Max number of simultaneous long positions in live trading
LIVE_BOOK_STRATEGIES     = ["momentum"]   # mean_reversion removed: never net-positive OOS in any mode
LIVE_REBALANCE_FREQ_DAYS = 7        # Weekly cadence — no orders placed more often than this

# ── Strategy Params ────────────────────────────────────────────────────────────
STRATEGY_PARAMS = {
    "momentum": {
        "lookback_long":  252,   # 12-month return (annualised)
        "lookback_short": 21,    # Skip last month (reversal filter)
        "top_n":          4,     # Long top N tokens
        "bottom_n":       4,     # Short bottom N tokens
        "hysteresis":     1,     # Extra ranks to cross before exiting (reduces churn)
    },
    "mean_reversion": {
        "zscore_window":  20,
        "entry_z":        2.0,
        "exit_z":         0.5,
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
        "lookback":        90,    # Rolling window for spread z-score and cointegration
        "top_pairs":       3,     # Number of best pairs to trade simultaneously
        "min_correlation": 0.60,  # Pearson |ρ| gate before running coint test
        "max_coint_pval":  0.05,  # Engle-Granger p-value threshold
        "reselect_freq":   5,     # Re-run pair selection every N bars
        "entry_z":         2.0,
        "exit_z":          0.0,
    },
    "exhaustion_fade": {
        "bb_window":          20,     # Bollinger Band lookback
        "bb_std":             2.5,    # Bollinger Band standard deviation multiplier
        "adx_period":         14,     # ADX smoothing period (Wilder)
        "adx_threshold":      30,     # ADX < threshold = ranging market
        "funding_threshold":  0.0005, # |funding rate| threshold for confirmation
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
    # ── Research / paper candidates (not in LIVE_BOOK_STRATEGIES) ─────────────
    "tsmom_volscaled": {
        "lookback":    252,    # 12-month trailing return for direction
        "vol_window":  126,    # 6-month realized vol estimation window (126→63 cut turnover ~60%)
        "min_vol":     0.05,   # 5% annualized vol floor (prevents ÷0 scaling)
    },
    "carry_neutral": {
        "funding_lookback": 7,     # days to smooth raw funding rate (weekly average)
    },
    "resid_momentum": {
        "ols_window":    63,        # rolling window for BTC-beta OLS (3 months)
        "mom_lookback":  252,       # cumulative residual lookback (12 months)
        "btc_proxy":    "BTCUSDT",  # factor to strip from each token
        "signal_smooth": 84,        # EMA span on normalized signal
        "weekly_update": True,      # only update signal every 5 bars; cuts turnover ~60%
    },
    "btc_dominance": {
        "btc_proxy":     "BTCUSDT", # BTC column in close/returns
        "smooth_window": 21,        # EMA span for dominance ratio smoothing
        "signal_smooth": 5,         # EMA span on final signal to reduce noise
    },
    "vol_spike_reversion": {
        "atr_period":      14,      # ATR lookback for spike threshold
        "spike_mult":       3.0,    # |1d return| > spike_mult × ATR → candidate bar
        "funding_confirm":  True,   # require funding sign flip (overcrowding confirmation)
        "hold_bars":        5,      # bars to hold position after entry (time stop)
        "signal_smooth":    3,      # EMA on signal to reduce whipsaw
    },
}

# ── Vol-targeting overlay ──────────────────────────────────────────────────────
# Applied post-hoc to any strategy's weights_history.  Replaces (not stacks
# with) the MIN_ANNUALIZED_VOL = 3% floor — that floor is irrelevant when the
# overlay is active since the overlay targets 15%.
VOL_TARGET_PARAMS = {
    "target_vol":  0.15,   # 15% annualized target portfolio vol
    "band":        0.20,   # ±20% corridor — only resize when outside [12%, 18%]
    "vol_window":  63,     # trailing days used to estimate realized vol
    "max_scale":   2.0,    # hard cap on the leverage-scaling factor
}

# ── Research gate thresholds ────────────────────────────────────────────────────
# A candidate strategy must clear ALL three to be eligible for live-book promotion.
# Promotion is manual — clearing the gate just flags the strategy as eligible.
RESEARCH_GATE = {
    "min_gross_sharpe_oos":   0.5,
    "min_net_sharpe_oos":     0.0,
    "max_annual_turnover_pct": 1000.0,
}

# ── Seasonality ────────────────────────────────────────────────────────────────
SEASONALITY_MIN_PERIODS  = 24   # Need at least 24 observations per season bucket
REGIME_MA_WINDOW         = 200  # BTC 200-day MA for bull/bear regime
STRATEGY_SELECT_METRIC   = "sharpe"  # sharpe | sortino | calmar
STRATEGY_RESELECT_DAYS   = 30   # Rebalance strategy weights at most monthly

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL   = "INFO"
LOG_ROTATION = "1 week"
