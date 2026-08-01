"""
db/connection.py — Thread-safe PostgreSQL connection pool.
DB_URL is read from the environment (set in Binance.env).

When PostgreSQL is unavailable (local dev without a DB), the pool init
silently fails and db_available() returns False so callers can degrade
gracefully instead of crashing.
"""

import os
from psycopg2 import pool as pg_pool
from loguru import logger

_pool: pg_pool.ThreadedConnectionPool | None = None
_db_available: bool | None = None   # None = not yet probed


def _get_pool() -> pg_pool.ThreadedConnectionPool | None:
    global _pool, _db_available
    if _db_available is False:
        return None
    if _pool is None or _pool.closed:
        url = os.environ.get(
            "DB_URL",
            "postgresql://sharonlau15:sharon@localhost:5432/crypto_algo",
        )
        try:
            _pool = pg_pool.ThreadedConnectionPool(1, 5, dsn=url)
            _db_available = True
            logger.info("PostgreSQL connection pool initialised (crypto_algo)")
        except Exception as e:
            _db_available = False
            logger.warning(f"PostgreSQL unavailable — running in no-DB mode: {e}")
            return None
    return _pool


def db_available() -> bool:
    """Return True if a PostgreSQL connection pool could be established."""
    if _db_available is None:
        _get_pool()   # probe on first call
    return bool(_db_available)


def get_conn():
    pool = _get_pool()
    if pool is None:
        raise ConnectionError("PostgreSQL is not available")
    return pool.getconn()


def put_conn(conn):
    try:
        pool = _get_pool()
        if pool is not None:
            pool.putconn(conn)
    except Exception:
        pass
