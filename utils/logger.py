"""
utils/logger.py  +  utils/reporting.py
=======================================
Logging configuration and console/file reporting helpers.
"""

# ── logger setup ───────────────────────────────────────────────────────────────
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
from config.settings import LOG_DIR, LOG_LEVEL, LOG_ROTATION


def setup_logger():
    logger.remove()
    logger.add(sys.stdout, level=LOG_LEVEL, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(
        LOG_DIR / "algo_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=LOG_ROTATION,
        retention="4 weeks",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
    )