"""
Debug VectorBT trade extraction - understand trades vs orders.
"""

import pickle
from pathlib import Path

try:
    import vectorbtpro as vbt
except ImportError:
    import vectorbt as vbt


def main():
    # Load signals
    signal_file = Path(__file__).parent / "signals" / "sp500_top10_sma_crossover.pkl"
    with open(signal_file, 'rb') as f:
        signal_set = pickle.load(f)

    asset_data = signal_set['assets']['AAPL']
    data = asset_data['data']
    signals = asset_data['signals']

    # Debug: Check column names
    print(f"Data columns: {list(data.columns)}")
    print(f"Signal columns: {list(signals.columns)}")
    print()

    # Run VectorBT
    pf = vbt.Portfolio.from_signals(
        close=data['close'],  # lowercase
        entries=signals['entry'],
        exits=signals['exit'],
        freq='1D',
        init_cash=100000,
        fees=0.0,
        slippage=0.0,
    )

    print('='*80)
    print('VECTORBT TRADES (complete round trips)')
    print('='*80)
    print(f'Count: {len(pf.trades.records)}')

    if hasattr(pf.trades, 'records_readable'):
        trades_df = pf.trades.records_readable
        print(f'Columns: {list(trades_df.columns)}')
        print()
        print('First 5 complete trades:')
        print(trades_df.head(5)[['Entry Timestamp', 'Exit Timestamp', 'Size', 'Avg Entry Price', 'Avg Exit Price']])
    print()

    print('='*80)
    print('VECTORBT ORDERS (individual buy/sell orders)')
    print('='*80)
    print(f'Count: {len(pf.orders.records)}')

    if hasattr(pf.orders, 'records_readable'):
        orders_df = pf.orders.records_readable
        print(f'Columns: {list(orders_df.columns)}')
        print()
        print('First 10 orders:')
        print(orders_df.head(10)[['Timestamp', 'Size', 'Price', 'Fees', 'Side']])
        print()

        # Count BUY vs SELL
        buy_count = (orders_df['Side'] == 'Buy').sum()
        sell_count = (orders_df['Side'] == 'Sell').sum()
        print(f'\nOrder breakdown:')
        print(f'  BUY orders:  {buy_count}')
        print(f'  SELL orders: {sell_count}')
        print(f'  Total:       {len(orders_df)}')


if __name__ == "__main__":
    main()
