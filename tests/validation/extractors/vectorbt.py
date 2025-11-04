"""VectorBT trade extractor.

Converts VectorBT portfolio trades to StandardTrade format.

VectorBT provides trades.records_readable with:
- 'Entry Index' (timestamp)
- 'Exit Index' (timestamp)
- 'Avg Entry Price'
- 'Avg Exit Price'
- 'Size'
- 'PnL'
- 'Entry Fees', 'Exit Fees'
"""

from typing import Any, List
import sys
from pathlib import Path

import pandas as pd

# Handle imports when run as script vs module
try:
    from ..core.trade import StandardTrade, infer_price_component
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.trade import StandardTrade, infer_price_component


def extract_vectorbt_trades(portfolio: Any, data: pd.DataFrame) -> List[StandardTrade]:
    """
    Extract trades from VectorBT portfolio.

    Args:
        portfolio: VectorBT Portfolio object
        data: Market data DataFrame (pandas) with OHLC columns

    Returns:
        List of StandardTrade objects

    Example:
        >>> portfolio = vbt.Portfolio.from_signals(...)
        >>> trades = extract_vectorbt_trades(portfolio, data)
    """
    trades_df = portfolio.trades.records_readable

    if trades_df.empty:
        return []

    standard_trades = []

    for idx, row in trades_df.iterrows():
        entry_ts = row['Entry Index']
        exit_ts = row.get('Exit Index')

        # Look up OHLC bars
        entry_bar = _get_bar_from_pandas(data, entry_ts) if entry_ts in data.index else {}
        exit_bar = _get_bar_from_pandas(data, exit_ts) if exit_ts and exit_ts in data.index else {}

        # Infer components
        entry_component = infer_price_component(row['Avg Entry Price'], entry_bar)
        exit_component = (
            infer_price_component(row.get('Avg Exit Price'), exit_bar)
            if exit_bar and pd.notna(row.get('Avg Exit Price'))
            else None
        )

        # Determine side
        side = 'long' if row.get('Direction', 'Long') == 'Long' else 'short'

        # Extract P&L components
        entry_fees = row.get('Entry Fees', 0.0)
        exit_fees = row.get('Exit Fees', 0.0)
        pnl = row.get('PnL', 0.0)
        gross_pnl = pnl + entry_fees + exit_fees if pd.notna(pnl) else None

        # Determine if closed
        is_closed = row.get('Status', 'Closed') == 'Closed'

        standard_trades.append(StandardTrade(
            trade_id=idx,
            platform='vectorbt',
            entry_timestamp=entry_ts.to_pydatetime() if hasattr(entry_ts, 'to_pydatetime') else entry_ts,
            entry_price=row['Avg Entry Price'],
            entry_price_component=entry_component,
            entry_bar_ohlc=entry_bar,
            exit_timestamp=(
                exit_ts.to_pydatetime() if exit_ts and hasattr(exit_ts, 'to_pydatetime')
                else exit_ts if is_closed
                else None
            ),
            exit_price=row.get('Avg Exit Price') if pd.notna(row.get('Avg Exit Price')) else None,
            exit_price_component=exit_component,
            exit_bar_ohlc=exit_bar if exit_bar else None,
            exit_reason='signal' if is_closed else None,
            symbol=data.columns.name if hasattr(data.columns, 'name') else 'UNKNOWN',
            quantity=abs(row['Size']),
            side=side,
            gross_pnl=gross_pnl,
            entry_commission=entry_fees,
            exit_commission=exit_fees,
            slippage=0.0,  # VectorBT handles slippage separately
            net_pnl=pnl if pd.notna(pnl) else None,
            signal_id=None,
            notes=f"VectorBT {row.get('Direction', 'Long')} trade",
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
        # For multi-asset data, take the first row (we can't determine which asset without more info)
        # This is a limitation - ideally we'd filter by symbol
        row = row.iloc[0]

    # Now row should be a Series
    if isinstance(row, pd.Series):
        # Standard case - extract OHLC from series
        bar = {
            'timestamp': timestamp,
            'open': row.get('open'),
            'high': row.get('high'),
            'low': row.get('low'),
            'close': row.get('close'),
            'volume': row.get('volume'),
        }
    else:
        # Fallback - shouldn't reach here
        bar = {}

    return bar
