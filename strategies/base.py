"""
strategies/base.py
==================
Abstract base class for all alpha strategies.

Every strategy must implement:
    generate_signals(close, returns, **kwargs) → pd.DataFrame

Signal conventions
------------------
    +1  = long signal (strong bullish)
     0  = flat / no position
    -1  = short signal (strong bearish)
    Values between -1 and +1 represent partial conviction.

The portfolio optimizer normalizes these into actual weights,
respecting all risk constraints from settings.py.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from loguru import logger


class BaseStrategy(ABC):
    """
    All strategies inherit from this class.

    Parameters
    ----------
    name   : str — unique identifier used in logging and results
    params : dict — strategy-specific parameters (from settings.STRATEGY_PARAMS)
    """

    def __init__(self, name: str, params: dict):
        self.name                 = name
        self.params               = params
        self._signal_cache: pd.DataFrame | None = None
        self.freeze_between_bands = False   # set True in subclass for band-freeze engine mode

    @abstractmethod
    def generate_signals(
        self,
        close:   pd.DataFrame,   # Wide: index=date, cols=symbols
        returns: pd.DataFrame,   # Log returns, same shape
        **kwargs,                 # Extra data (funding rates, sentiment, etc.)
    ) -> pd.DataFrame:
        """
        Compute signal matrix.

        Returns
        -------
        pd.DataFrame — same shape as `close`.
            Values in [-1, +1]. NaN = no view (treated as 0 by optimizer).
        """
        ...

    def run(
        self,
        close:   pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Wrapper around generate_signals with validation and caching.
        Call this from the backtest engine, not generate_signals directly.
        """
        logger.debug(f"Running strategy: {self.name}")
        signals = self.generate_signals(close, returns, **kwargs)

        # Validate output shape
        assert signals.shape == close.shape, (
            f"{self.name}: signal shape {signals.shape} ≠ close shape {close.shape}"
        )
        # Clip to [-1, 1]
        signals = signals.clip(-1, 1)
        # Forward-fill NaNs but never look ahead
        signals = signals.shift(1)   # T+1 execution: signal from T used at T+1
        self._signal_cache = signals
        return signals

    def get_last_signal(self) -> pd.Series | None:
        """Return the most recent row of signals (for live trading)."""
        if self._signal_cache is None:
            return None
        return self._signal_cache.iloc[-1]

    def __repr__(self) -> str:
        return f"<Strategy: {self.name}>"