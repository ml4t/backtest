"""Zipline-reloaded trade extractor.

Converts Zipline performance DataFrame to StandardTrade format.

Zipline provides transaction records through the performance DataFrame
returned by run_algorithm(). Transactions need to be matched into
complete trades (entry + exit).
"""

from typing import Any, List
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Handle imports when run as script vs module
try:
    from ..core.trade import StandardTrade, infer_price_component
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.trade import StandardTrade, infer_price_component


def extract_zipline_trades(
    perf: pd.DataFrame,
    data: pd.DataFrame
) -> List[StandardTrade]:
    """
    Extract trades from Zipline performance DataFrame.

    Args:
        perf: Performance DataFrame from run_algorithm()
              Contains 'transactions' column with transaction records
        data: Market data DataFrame (pandas) with OHLC columns

    Returns:
        List of StandardTrade objects

    Note:
        Zipline returns transactions (individual buy/sell orders).
        We match these into complete trades using FIFO logic.

    Example:
        >>> perf = zipline.run_algorithm(...)
        >>> trades = extract_zipline_trades(perf, data)
    """
    # Extract all transactions from performance DataFrame
    all_transactions = []

    for date, row in perf.iterrows():
        transactions = row.get('transactions', [])
        if transactions:
            for txn in transactions:
                all_transactions.append({
                    'dt': txn.get('dt', date),
                    'sid': txn.get('sid'),
                    'amount': txn.get('amount', 0),
                    'price': txn.get('price', 0),
                    'commission': txn.get('commission', 0),
                })

    if not all_transactions:
        return []

    # Sort by datetime
    all_transactions.sort(key=lambda x: x['dt'])

    # Match transactions into complete trades (FIFO)
    standard_trades = []
    open_positions: dict[Any, list[dict]] = {}

    for txn in all_transactions:
        sid = txn['sid']
        amount = txn['amount']
        price = txn['price']
        dt = txn['dt']
        commission = txn['commission']

        if amount > 0:  # Buy
            # Open position
            if sid not in open_positions:
                open_positions[sid] = []

            # Look up OHLC at entry
            entry_bar = _get_bar_from_pandas(data, dt) if dt in data.index else {}
            entry_component = infer_price_component(price, entry_bar)

            open_positions[sid].append({
                'entry_time': dt,
                'entry_price': price,
                'entry_component': entry_component,
                'entry_bar': entry_bar,
                'quantity': abs(amount),
                'entry_commission': commission,
            })

        elif amount < 0 and sid in open_positions and open_positions[sid]:  # Sell
            # Close position (FIFO)
            position = open_positions[sid].pop(0)

            # Look up OHLC at exit
            exit_bar = _get_bar_from_pandas(data, dt) if dt in data.index else {}
            exit_component = infer_price_component(price, exit_bar)

            # Calculate P&L
            quantity = position['quantity']
            gross_pnl = (price - position['entry_price']) * quantity
            entry_comm = position['entry_commission'] or 0.0
            exit_comm = commission or 0.0
            total_commission = entry_comm + exit_comm
            net_pnl = gross_pnl - total_commission

            standard_trades.append(StandardTrade(
                trade_id=len(standard_trades),
                platform='zipline',
                entry_timestamp=(
                    position['entry_time'].to_pydatetime()
                    if hasattr(position['entry_time'], 'to_pydatetime')
                    else position['entry_time']
                ),
                entry_price=position['entry_price'],
                entry_price_component=position['entry_component'],
                entry_bar_ohlc=position['entry_bar'],
                exit_timestamp=(
                    dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt
                ),
                exit_price=price,
                exit_price_component=exit_component,
                exit_bar_ohlc=exit_bar,
                exit_reason='signal',
                symbol=str(sid) if sid else 'UNKNOWN',
                quantity=quantity,
                side='long',  # TODO: Handle shorts
                gross_pnl=gross_pnl,
                entry_commission=entry_comm,
                exit_commission=exit_comm,
                slippage=0.0,  # Zipline handles slippage in price
                net_pnl=net_pnl,
                signal_id=None,
                notes=f"Zipline long trade for {sid}",
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
