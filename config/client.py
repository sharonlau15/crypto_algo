"""
config/client.py
================
Dual Binance client: real API for data ingestion (read-only),
demo trading client for order execution (virtual money, real prices).

Usage:
    from config.client import real_client, demo_client, get_client

Demo Trading vs Testnet
-----------------------
Demo trading uses the real Binance API endpoint (api.binance.com) with
a demo account — real-time prices, virtual funds. Keys are obtained from
your Binance account under the Paper Trading / Demo Trading section.

Testnet (testnet.binance.vision) is a separate server with fake data and
separate keys. Demo trading is more realistic because it uses live prices
and a real matching engine against the live order book.
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

# ── Demo Trading — ORDER EXECUTION with virtual USDT, real-time prices ────────
# Does NOT use testnet=True — demo trading hits the real Binance API endpoint
# with demo account credentials, giving accurate fills at live market prices.
demo_client = Client(DEMO_API_KEY, DEMO_API_SECRET)

logger.info("✅ Connected to Binance (real) for market data")
logger.info("✅ Connected to Binance Demo Trading for simulated order execution")


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
