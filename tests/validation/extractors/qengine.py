"""QEngine trade extractor.

Converts qengine broker results to StandardTrade format.

Challenge: qengine returns individual orders (BUY/SELL),
must match into complete trades (entry + exit).
"""

from typing import Any, List
import sys
from pathlib import Path

import polars as pl

# Handle imports when run as script vs module
try:
    from ..core.trade import StandardTrade, get_bar_at_timestamp, infer_price_component
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.trade import StandardTrade, get_bar_at_timestamp, infer_price_component


def extract_qengine_trades(results: dict[str, Any], data: pl.DataFrame) -> List[StandardTrade]:
    """
    Extract trades from qengine broker results.

    Args:
        results: Dict from BacktestEngine.run() with 'trades' key
        data: Market data DataFrame with timestamp, open, high, low, close, volume

    Returns:
        List of StandardTrade objects

    Example:
        >>> results = engine.run()
        >>> trades = extract_qengine_trades(results, data)
        >>> for trade in trades:
        ...     print(f"{trade.entry_timestamp} -> {trade.exit_timestamp}: ${trade.net_pnl:.2f}")
    """
    trades_df = results.get('trades')
    if trades_df is None or trades_df.is_empty():
        return []

    standard_trades = []
    open_positions: dict[str, list[dict[str, Any]]] = {}

    # Sort by filled_time to process chronologically
    for row in trades_df.sort('filled_time').iter_rows(named=True):
        asset_id = row['asset_id']
        side = row['side'].upper()  # Normalize to uppercase

        if side == 'BUY':
            # Open position
            if asset_id not in open_positions:
                open_positions[asset_id] = []

            # Look up OHLC at entry
            entry_bar = get_bar_at_timestamp(data, row['filled_time'])
            entry_component = infer_price_component(row['price'], entry_bar)

            open_positions[asset_id].append({
                'entry_timestamp': row['filled_time'],
                'entry_price': row['price'],
                'entry_component': entry_component,
                'entry_bar': entry_bar,
                'quantity': row['quantity'],
                'entry_commission': row['commission'],
                'order_id': row.get('order_id'),
            })

        elif side == 'SELL' and asset_id in open_positions and open_positions[asset_id]:
            # Close position (FIFO matching)
            position = open_positions[asset_id].pop(0)

            # Look up OHLC at exit
            exit_bar = get_bar_at_timestamp(data, row['filled_time'])
            exit_component = infer_price_component(row['price'], exit_bar)

            # Calculate P&L
            gross_pnl = (row['price'] - position['entry_price']) * position['quantity']
            total_commission = position['entry_commission'] + row['commission']
            net_pnl = gross_pnl - total_commission

            standard_trades.append(StandardTrade(
                trade_id=len(standard_trades),
                platform='qengine',
                entry_timestamp=position['entry_timestamp'],
                entry_price=position['entry_price'],
                entry_price_component=position['entry_component'],
                entry_bar_ohlc=position['entry_bar'],
                exit_timestamp=row['filled_time'],
                exit_price=row['price'],
                exit_price_component=exit_component,
                exit_bar_ohlc=exit_bar,
                exit_reason='signal',  # From signal-driven strategy
                symbol=asset_id,
                quantity=position['quantity'],
                side='long',  # TODO: Handle shorts
                gross_pnl=gross_pnl,
                entry_commission=position['entry_commission'],
                exit_commission=row['commission'],
                slippage=0.0,  # TODO: Extract from order if available
                net_pnl=net_pnl,
                signal_id=None,
                notes=f"Entry order: {position.get('order_id')}, Exit order: {row.get('order_id')}",
            ))

    return standard_trades
