"""
portfolio/risk_manager.py
=========================
Position-level risk management with stop loss, take profit, and trailing stops.

Features:
- Hard stop loss (exit if loss exceeds threshold)
- Take profit targets (exit if profit target hit)
- Trailing stops (dynamic based on peak price)
- Position entry tracking with historical prices
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from loguru import logger


@dataclass
class Position:
    """Track a single open position with entry price and stops."""
    symbol: str
    entry_price: float
    entry_date: int  # day index
    quantity: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    peak_price: float = field(init=False)
    
    def __post_init__(self):
        self.peak_price = self.entry_price


class PositionRiskManager:
    """
    Manages stop loss, take profit, and trailing stops for individual positions.
    
    Parameters
    ----------
    stop_loss_pct : float
        Hard stop loss as % of entry price (e.g., 0.02 = 2% stop loss)
    take_profit_pct : float
        Take profit target as % of entry price (e.g., 0.03 = 3% take profit)
    trailing_stop_pct : float or None
        Trailing stop as % of peak price (e.g., 0.05 = 5% trailing stop)
        If None, no trailing stop.
    use_trailing : bool
        Whether to use trailing stop for longs (shorts use hard stop)
    """
    
    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
        trailing_stop_pct: Optional[float] = None,
        use_trailing: bool = True,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.use_trailing = use_trailing
        
        self.positions: Dict[str, Position] = {}  # symbol → Position
        self.exit_history = []  # Track exits for reporting
    
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        day_index: int,
        quantity: float,
    ) -> Position:
        """Open a new position with stops calculated from entry price."""
        
        # Calculate stop prices
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)
        
        pos = Position(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=day_index,
            quantity=quantity,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
        )
        
        self.positions[symbol] = pos
        logger.debug(
            f"[RISK] Position opened: {symbol} @ ${entry_price:.4f} | "
            f"SL: ${stop_loss:.4f} (-{self.stop_loss_pct*100:.1f}%) | "
            f"TP: ${take_profit:.4f} (+{self.take_profit_pct*100:.1f}%)"
        )
        return pos
    
    def check_exits(
        self,
        current_prices: Dict[str, float],
        day_index: int,
    ) -> Dict[str, Tuple[str, float]]:
        """
        Check all positions for stop loss / take profit triggers.
        
        Returns
        -------
        dict: symbol → (exit_reason, exit_price)
            Only includes positions that should be exited
        """
        exits = {}
        
        for symbol, pos in list(self.positions.items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Update peak for trailing stop
            if current_price > pos.peak_price:
                pos.peak_price = current_price
            
            # Check take profit first (most favorable)
            if current_price >= pos.take_profit_price:
                exits[symbol] = ("TAKE_PROFIT", pos.take_profit_price)
                self.exit_history.append({
                    "symbol": symbol,
                    "entry_price": pos.entry_price,
                    "exit_price": pos.take_profit_price,
                    "exit_reason": "TAKE_PROFIT",
                    "entry_date": pos.entry_date,
                    "exit_date": day_index,
                    "pnl_pct": ((pos.take_profit_price - pos.entry_price) / pos.entry_price),
                })
                continue
            
            # Check trailing stop (if enabled and we have gains)
            if self.use_trailing and self.trailing_stop_pct and current_price > pos.entry_price:
                trailing_stop = pos.peak_price * (1 - self.trailing_stop_pct)
                if current_price <= trailing_stop:
                    exits[symbol] = ("TRAILING_STOP", current_price)
                    self.exit_history.append({
                        "symbol": symbol,
                        "entry_price": pos.entry_price,
                        "exit_price": current_price,
                        "exit_reason": "TRAILING_STOP",
                        "entry_date": pos.entry_date,
                        "exit_date": day_index,
                        "pnl_pct": ((current_price - pos.entry_price) / pos.entry_price),
                    })
                    continue
            
            # Check hard stop loss
            if current_price <= pos.stop_loss_price:
                exits[symbol] = ("STOP_LOSS", pos.stop_loss_price)
                self.exit_history.append({
                    "symbol": symbol,
                    "entry_price": pos.entry_price,
                    "exit_price": pos.stop_loss_price,
                    "exit_reason": "STOP_LOSS",
                    "entry_date": pos.entry_date,
                    "exit_date": day_index,
                    "pnl_pct": ((pos.stop_loss_price - pos.entry_price) / pos.entry_price),
                })
        
        # Remove exited positions
        for symbol in exits:
            del self.positions[symbol]
        
        return exits
    
    def close_position(self, symbol: str, exit_price: float, day_index: int):
        """Manually close a position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
            self.exit_history.append({
                "symbol": symbol,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "exit_reason": "REBALANCE",
                "entry_date": pos.entry_date,
                "exit_date": day_index,
                "pnl_pct": pnl_pct,
            })
            del self.positions[symbol]
    
    def get_active_symbols(self) -> list[str]:
        """Return list of symbols with open positions."""
        return list(self.positions.keys())
    
    def get_position_count(self) -> int:
        """Return number of open positions."""
        return len(self.positions)
    
    def get_exit_report(self) -> pd.DataFrame:
        """Return DataFrame of all exits (for analysis)."""
        if not self.exit_history:
            return pd.DataFrame()
        return pd.DataFrame(self.exit_history)
    
    def get_exit_stats(self) -> dict:
        """Compute aggregate exit statistics."""
        if not self.exit_history:
            return {}
        
        df = pd.DataFrame(self.exit_history)
        
        return {
            "total_exits": len(df),
            "stop_loss_exits": (df["exit_reason"] == "STOP_LOSS").sum(),
            "take_profit_exits": (df["exit_reason"] == "TAKE_PROFIT").sum(),
            "trailing_stop_exits": (df["exit_reason"] == "TRAILING_STOP").sum(),
            "rebalance_exits": (df["exit_reason"] == "REBALANCE").sum(),
            "avg_exit_pnl_pct": df["pnl_pct"].mean(),
            "median_exit_pnl_pct": df["pnl_pct"].median(),
            "winning_exits": (df["pnl_pct"] > 0).sum(),
            "losing_exits": (df["pnl_pct"] < 0).sum(),
            "best_exit": df["pnl_pct"].max(),
            "worst_exit": df["pnl_pct"].min(),
        }
