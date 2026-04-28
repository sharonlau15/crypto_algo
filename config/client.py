"""
config/client.py
================
Dual Binance client: real API for data ingestion (read-only),
testnet client for order execution (paper money, real matching engine).

Usage:
    from config.client import real_client, testnet_client, get_client
"""

import logging
from binance.client import Client
from config.settings import (
    API_KEY, API_SECRET,
    TESTNET_API_KEY, TESTNET_API_SECRET,
    PAPER_TRADING,
)

logger = logging.getLogger(__name__)

# ── Real Binance — DATA ONLY, never place orders here ─────────────────────────
real_client = Client(API_KEY, API_SECRET)

# ── Testnet — ORDER EXECUTION with fake USDT ──────────────────────────────────
testnet_client = Client(
    TESTNET_API_KEY,
    TESTNET_API_SECRET,
    testnet=True,
)

logger.info("✅ Connected to Binance (real) for market data")
logger.info("✅ Connected to Binance Testnet for simulated trading")


def get_client(for_trading: bool = False) -> Client:
    """
    Return the appropriate client.

    Parameters
    ----------
    for_trading : bool
        True  → testnet client (order placement, account queries)
        False → real client   (klines, orderbook, ticker data)

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
        return testnet_client
    return real_client


def check_connectivity() -> dict:
    """Ping both endpoints and return latency info."""
    import time
    results = {}

    t0 = time.time()
    real_client.ping()
    results["real_latency_ms"] = round((time.time() - t0) * 1000, 1)

    t0 = time.time()
    testnet_client.ping()
    results["testnet_latency_ms"] = round((time.time() - t0) * 1000, 1)

    logger.info(f"Connectivity: {results}")
    return results