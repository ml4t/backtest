"""Backtrader trade extractor.

Converts Backtrader trade records to StandardTrade format.

Backtrader tracks trades through:
- Strategy trade notifications (on_trade callback)
- TradeAnalyzer results
- Manual tracking of buy/sell orders
"""

from typing import Any, List
import sys
from pathlib import Path
from datetime import datetime
import pytz

import pandas as pd

# Handle imports when run as script vs module
try:
    from ..core.trade import StandardTrade, infer_price_component
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.trade import StandardTrade, infer_price_component


def extract_backtrader_trades(
    trades_list: List[dict],
    data: pd.DataFrame
) -> List[StandardTrade]:
    """
    Extract trades from Backtrader trade records.

    Args:
        trades_list: List of trade dicts collected from strategy.
                     Each dict should have:
                     - 'entry_time': datetime
                     - 'exit_time': datetime
                     - 'entry_price': float
                     - 'exit_price': float
                     - 'size': float (positive for long, negative for short)
                     - 'pnl': float (gross P&L)
                     - 'commission': float (total commission)
        data: Market data DataFrame (pandas) with OHLC columns

    Returns:
        List of StandardTrade objects

    Example:
        >>> # In Backtrader strategy, collect trades:
        >>> def notify_trade(self, trade):
        ...     if trade.isclosed:
        ...         self.trades.append({
        ...             'entry_time': bt.num2date(trade.dtopen),
        ...             'exit_time': bt.num2date(trade.dtclose),
        ...             'entry_price': trade.price,
        ...             'exit_price': trade.pnl / trade.size + trade.price,
        ...             'size': trade.size,
        ...             'pnl': trade.pnl,
        ...             'commission': trade.commission,
        ...         })
        >>>
        >>> # Then extract:
        >>> trades = extract_backtrader_trades(strategy.trades, data)
    """
    if not trades_list:
        return []

    standard_trades = []

    for idx, trade_dict in enumerate(trades_list):
        entry_ts = trade_dict['entry_time']
        exit_ts = trade_dict.get('exit_time')

        # Ensure timestamps are datetime objects
        if isinstance(entry_ts, str):
            entry_ts = pd.to_datetime(entry_ts)
        if isinstance(exit_ts, str):
            exit_ts = pd.to_datetime(exit_ts)

        # CRITICAL: Make timestamps timezone-aware (UTC) if they're naive
        # Backtrader's bt.num2date() returns timezone-naive datetimes
        if entry_ts and not entry_ts.tzinfo:
            entry_ts = entry_ts.replace(tzinfo=pytz.UTC)
        if exit_ts and not exit_ts.tzinfo:
            exit_ts = exit_ts.replace(tzinfo=pytz.UTC)

        # Look up OHLC bars
        entry_bar = _get_bar_from_pandas(data, entry_ts) if entry_ts in data.index else {}
        exit_bar = _get_bar_from_pandas(data, exit_ts) if exit_ts and exit_ts in data.index else {}

        # Infer components
        entry_component = infer_price_component(trade_dict['entry_price'], entry_bar)
        exit_component = (
            infer_price_component(trade_dict.get('exit_price'), exit_bar)
            if exit_bar and trade_dict.get('exit_price')
            else None
        )

        # Determine side
        size = trade_dict['size']
        side = 'long' if size > 0 else 'short'

        # Extract economics
        gross_pnl = trade_dict.get('pnl', 0.0)
        commission = trade_dict.get('commission', 0.0)

        # Backtrader typically includes commission in pnl, so net_pnl = gross_pnl
        # But we'll calculate it explicitly for consistency
        net_pnl = gross_pnl - commission if gross_pnl is not None else None

        # Split commission between entry and exit (approximate 50/50)
        entry_commission = commission / 2 if commission else 0.0
        exit_commission = commission / 2 if commission else 0.0

        # Check if closed
        is_closed = exit_ts is not None

        standard_trades.append(StandardTrade(
            trade_id=idx,
            platform='backtrader',
            entry_timestamp=entry_ts.to_pydatetime() if hasattr(entry_ts, 'to_pydatetime') else entry_ts,
            entry_price=trade_dict['entry_price'],
            entry_price_component=entry_component,
            entry_bar_ohlc=entry_bar,
            exit_timestamp=(
                exit_ts.to_pydatetime() if exit_ts and hasattr(exit_ts, 'to_pydatetime')
                else exit_ts if is_closed
                else None
            ),
            exit_price=trade_dict.get('exit_price'),
            exit_price_component=exit_component,
            exit_bar_ohlc=exit_bar if exit_bar else None,
            exit_reason='signal' if is_closed else None,
            symbol=data.columns.name if hasattr(data.columns, 'name') else 'UNKNOWN',
            quantity=abs(size),
            side=side,
            gross_pnl=gross_pnl,
            entry_commission=entry_commission,
            exit_commission=exit_commission,
            slippage=0.0,  # Backtrader handles slippage separately
            net_pnl=net_pnl,
            signal_id=None,
            notes=f"Backtrader {side} trade",
        ))

    return standard_trades


def _get_bar_from_pandas(data: pd.DataFrame, timestamp: pd.Timestamp) -> dict[str, Any]:
    """
    Extract OHLC bar from pandas DataFrame at timestamp.

    Args:
        data: DataFrame with OHLC columns
        timestamp: Target timestamp

    Returns:
        Dict with OHLC data
    """
    if timestamp not in data.index:
        return {}

    row = data.loc[timestamp]

    # Handle DataFrame (multiple rows with same timestamp - multi-asset case)
    if isinstance(row, pd.DataFrame):
        # For multi-asset data, take the first row
        row = row.iloc[0]

    # Now row should be a Series
    if isinstance(row, pd.Series):
        bar = {
            'timestamp': timestamp,
            'open': row.get('open'),
            'high': row.get('high'),
            'low': row.get('low'),
            'close': row.get('close'),
            'volume': row.get('volume'),
        }
    else:
        # Fallback
        bar = {}

    return bar
