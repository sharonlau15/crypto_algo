"""
data/ingestion.py
=================
Fetch OHLCV klines and funding rates from Binance.
All results are cached as Parquet files to avoid redundant API calls.
Cache is invalidated after CACHE_EXPIRY_HOURS (see settings).

Public API
----------
    get_ohlcv(symbol, interval, start, end)  → pd.DataFrame
    get_universe_ohlcv(interval, start, end) → dict[str, pd.DataFrame]
    get_funding_rates(symbol, limit)         → pd.DataFrame
    get_fear_greed_index(limit)              → pd.DataFrame
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from config.client import real_client
from config.settings import (
    UNIVERSE, KLINE_INTERVAL, BACKTEST_START,
    BACKTEST_END, DATA_DIR, CACHE_EXPIRY_HOURS,
    MIN_DAILY_VOLUME_USDT,
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _cache_path(symbol: str, interval: str) -> Path:
    return DATA_DIR / f"{symbol}_{interval}.parquet"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_EXPIRY_HOURS


def _klines_to_df(raw: list) -> pd.DataFrame:
    """Convert Binance kline list to a clean OHLCV DataFrame."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
    df = df.set_index("date")[["open", "high", "low", "close", "volume", "quote_volume"]]
    df = df.apply(pd.to_numeric)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


# ── Public functions ───────────────────────────────────────────────────────────

def get_ohlcv(
    symbol: str,
    interval: str = KLINE_INTERVAL,
    start: str = BACKTEST_START,
    end: str | None = BACKTEST_END,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV for a single symbol.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and columns:
        open, high, low, close, volume, quote_volume
    """
    path = _cache_path(symbol, interval)

    if use_cache and _cache_is_fresh(path):
        logger.debug(f"Cache hit: {symbol} {interval}")
        return pd.read_parquet(path)

    logger.info(f"Fetching {symbol} {interval} from {start}")

    end_str = end or datetime.utcnow().strftime("%Y-%m-%d")
    raw = real_client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start,
        end_str=end_str,
    )

    if not raw:
        logger.warning(f"No data returned for {symbol}")
        return pd.DataFrame()

    df = _klines_to_df(raw)

    # Filter low-volume days (data quality guard)
    df = df[df["quote_volume"] >= MIN_DAILY_VOLUME_USDT]

    df.to_parquet(path)
    logger.success(f"Cached {symbol}: {len(df)} bars → {path.name}")
    return df


def get_universe_ohlcv(
    interval: str = KLINE_INTERVAL,
    start: str = BACKTEST_START,
    end: str | None = BACKTEST_END,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for the entire universe in parallel.

    Returns
    -------
    dict mapping symbol → DataFrame
    """
    universe_data = {}
    for sym in tqdm(UNIVERSE, desc="Fetching universe OHLCV"):
        try:
            df = get_ohlcv(sym, interval, start, end, use_cache)
            if not df.empty:
                universe_data[sym] = df
        except Exception as e:
            logger.error(f"Failed to fetch {sym}: {e}")
        time.sleep(0.1)  # polite rate limit

    logger.info(f"Universe loaded: {len(universe_data)}/{len(UNIVERSE)} symbols")
    return universe_data


def build_close_matrix(universe_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Stack close prices into a single wide DataFrame.

    Returns
    -------
    pd.DataFrame: index=date, columns=symbols, values=close price
    """
    closes = {sym: df["close"] for sym, df in universe_data.items()}
    matrix = pd.DataFrame(closes)
    matrix = matrix.sort_index().ffill().dropna(how="all")
    return matrix


def build_return_matrix(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns from close matrix."""
    return np.log(close_matrix / close_matrix.shift(1)).dropna(how="all")


def get_funding_rates(symbol: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch 8-hour perpetual funding rates from Binance.
    Used as a carry signal proxy.

    Returns
    -------
    pd.DataFrame: index=datetime, columns=[symbol, funding_rate]
    """
    try:
        raw = real_client.futures_funding_rate(symbol=symbol, limit=limit)
        df = pd.DataFrame(raw)
        df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df.set_index("datetime")[["funding_rate"]]
        df.columns = [symbol]
        return df
    except Exception as e:
        logger.warning(f"Funding rate unavailable for {symbol}: {e}")
        return pd.DataFrame()


def get_fear_greed_index(limit: int = 365) -> pd.DataFrame:
    """
    Fetch the Crypto Fear & Greed Index from alternative.me (free, no auth).

    Returns
    -------
    pd.DataFrame: index=date, columns=[value, classification]
        value: 0 (extreme fear) → 100 (extreme greed)
    """
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.normalize()
        df["value"] = df["value"].astype(int)
        df = df.set_index("date")[["value", "value_classification"]]
        df.columns = ["fg_value", "fg_class"]
        df = df.sort_index()
        return df
    except Exception as e:
        logger.warning(f"Fear & Greed index unavailable: {e}")
        return pd.DataFrame()


def get_universe_funding_rates() -> pd.DataFrame:
    """
    Fetch and merge daily-averaged funding rates for all universe tokens.
    Only works for perpetual futures pairs (USDT-margined).

    Returns
    -------
    pd.DataFrame: index=date, columns=symbols
    """
    frames = []
    for sym in tqdm(UNIVERSE, desc="Fetching funding rates"):
        df = get_funding_rates(sym)
        if not df.empty:
            # Resample 8h → daily mean
            daily = df.resample("1D").mean()
            frames.append(daily)
        time.sleep(0.1)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index().ffill()
    return combined