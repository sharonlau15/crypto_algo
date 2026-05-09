"""
config/client.py
================
Dual Binance client: real API for data ingestion (read-only),
demo trading client for futures order execution (virtual money).

Endpoint routing
----------------
  real_client  → api.binance.com / fapi.binance.com   (live, read-only)
  demo_client  → testnet.binancefuture.com (futures)  (paper trading)

Binance Demo Trading Futures uses testnet.binancefuture.com as its
execution endpoint — confirmed via API diagnostics (code -2015 on
fapi.binance.com, success on testnet.binancefuture.com with demo keys).
Setting testnet=True on the demo client routes futures calls correctly.
Spot market data always comes from real_client so the testnet spot URL
on demo_client is never used.
"""

import logging
from binance.client import Client
from config.settings import (
    API_KEY, API_SECRET,
    DEMO_API_KEY, DEMO_API_SECRET,
    PAPER_TRADING,
)

logger = logging.getLogger(__name__)

# ── Real Binance — DATA ONLY, never place orders here ─────────────────────────
real_client = Client(API_KEY, API_SECRET)

# ── Demo Futures — ORDER EXECUTION via testnet.binancefuture.com ──────────────
# testnet=True routes futures calls to testnet.binancefuture.com which is the
# correct endpoint for Binance Demo Trading Futures (paper money, real prices).
demo_client = Client(DEMO_API_KEY, DEMO_API_SECRET, testnet=True)

logger.info("✅ Connected to Binance (real) for market data")
logger.info("✅ Connected to Binance Demo Futures (testnet.binancefuture.com)")


def get_client(for_trading: bool = False) -> Client:
    """
    Return the appropriate client.

    Parameters
    ----------
    for_trading : bool
        True  → demo trading client (order placement, account queries)
        False → real client        (klines, orderbook, ticker data)

    The PAPER_TRADING guard ensures live orders are never placed
    unless explicitly disabled in settings.py.
    """
    if for_trading:
        if not PAPER_TRADING:
            logger.warning(
                "⚠️  PAPER_TRADING=False — returning LIVE client. "
                "Real funds at risk. Confirm this is intentional."
            )
            return real_client
        return demo_client
    return real_client


def check_connectivity() -> dict:
    """Ping both endpoints and return latency info."""
    import time
    results = {}

    t0 = time.time()
    real_client.ping()
    results["real_latency_ms"] = round((time.time() - t0) * 1000, 1)

    t0 = time.time()
    demo_client.ping()
    results["demo_latency_ms"] = round((time.time() - t0) * 1000, 1)

    logger.info(f"Connectivity: {results}")
    return results
