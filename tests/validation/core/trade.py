"""
Standard trade representation for cross-platform comparison.

This module provides a platform-independent trade format that all
backtesting platforms must convert to for comparison.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl


@dataclass
class StandardTrade:
    """
    Platform-independent trade representation.

    Every platform (ml4t.backtest, VectorBT, Backtrader, Zipline) must
    convert its native trade format to this standard for comparison.
    """

    # Identity
    trade_id: int                     # Sequential ID within platform
    platform: str                     # 'ml4t.backtest', 'vectorbt', 'backtrader', 'zipline'

    # Entry
    entry_timestamp: datetime         # When did we enter?
    entry_price: float               # At what price?
    entry_price_component: str       # 'open', 'close', 'high', 'low', 'unknown'
    entry_bar_ohlc: dict[str, Any]  # Complete OHLC of entry bar

    # Exit (None if position still open)
    exit_timestamp: datetime | None   # When did we exit?
    exit_price: float | None         # At what price?
    exit_price_component: str | None  # 'open', 'close', 'high', 'low', 'unknown'
    exit_bar_ohlc: dict[str, Any] | None  # Complete OHLC of exit bar
    exit_reason: str | None          # 'signal', 'stop_loss', 'take_profit', 'trailing_stop'

    # Trade details
    symbol: str
    quantity: float
    side: str                        # 'long' or 'short'

    # Economics
    gross_pnl: float | None          # Price difference * quantity (before costs)
    entry_commission: float          # Commission paid on entry
    exit_commission: float           # Commission paid on exit
    slippage: float                  # Total slippage (entry + exit)
    net_pnl: float | None            # After all costs

    # Metadata
    signal_id: str | None = None     # Link back to originating signal
    notes: str | None = None         # Platform-specific notes

    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_timestamp is not None

    def total_cost(self) -> float:
        """Total transaction costs."""
        return self.entry_commission + self.exit_commission + self.slippage

    def __str__(self) -> str:
        """Human-readable representation."""
        exit_str = f"→ {self.exit_timestamp.strftime('%Y-%m-%d %H:%M')}" if self.exit_timestamp else "→ OPEN"
        pnl_str = f"${self.net_pnl:.2f}" if self.net_pnl is not None else "N/A"
        exit_price_str = f"${self.exit_price:.2f}" if self.exit_price is not None else "N/A"
        return (f"{self.platform} Trade #{self.trade_id}: "
                f"{self.entry_timestamp.strftime('%Y-%m-%d %H:%M')} {exit_str} | "
                f"{self.side.upper()} {self.quantity} {self.symbol} @ "
                f"${self.entry_price:.2f}→{exit_price_str} | "
                f"PnL: {pnl_str}")


def get_bar_at_timestamp(data: pl.DataFrame, timestamp: datetime) -> dict[str, Any]:
    """
    Get OHLC bar at specific timestamp.

    Args:
        data: Polars DataFrame with timestamp, open, high, low, close, volume
        timestamp: Target timestamp

    Returns:
        Dict with OHLC data, or empty dict if not found
    """
    bar = data.filter(pl.col('timestamp') == timestamp)
    if len(bar) == 0:
        return {}

    # Convert first row to dict
    row_dict = bar.row(0, named=True)
    return {
        'timestamp': row_dict['timestamp'],
        'open': row_dict['open'],
        'high': row_dict['high'],
        'low': row_dict['low'],
        'close': row_dict['close'],
        'volume': row_dict.get('volume'),
    }


def infer_price_component(
    price: float,
    bar: dict[str, Any],
    tolerance: float = 0.01
) -> str:
    """
    Infer which OHLC component was used based on price.

    Compares the execution price to OHLC components and returns
    the closest match.

    Args:
        price: The actual execution price
        bar: OHLC bar dict from get_bar_at_timestamp()
        tolerance: Price match tolerance as percentage (0.01 = 1%)

    Returns:
        'open', 'close', 'high', 'low', or 'unknown'

    Examples:
        >>> bar = {'open': 73.50, 'high': 74.00, 'low': 73.00, 'close': 73.85}
        >>> infer_price_component(73.50, bar)
        'open'
        >>> infer_price_component(73.85, bar)
        'close'
        >>> infer_price_component(100.00, bar)  # No match
        'unknown'
    """
    if not bar or price is None:
        return 'unknown'

    components = {
        'open': bar.get('open'),
        'high': bar.get('high'),
        'low': bar.get('low'),
        'close': bar.get('close'),
    }

    # Check exact match first (within 0.1 cent)
    for name, value in components.items():
        if value is not None and abs(price - value) < 0.001:
            return name

    # Check within tolerance
    best_match = None
    best_diff = float('inf')

    for name, value in components.items():
        if value is None or value == 0:
            continue

        pct_diff = abs(price - value) / value
        if pct_diff < tolerance and pct_diff < best_diff:
            best_match = name
            best_diff = pct_diff

    return best_match if best_match else 'unknown'
