#!/usr/bin/env python3
"""
Analyze why VectorBT shows different return than Zipline despite matching signals.
"""

import pandas as pd
from tests.validation.data_loader import UniversalDataLoader

def main():
    print("=" * 80)
    print("TRADE-LEVEL RECONCILIATION")
    print("=" * 80)

    # Load data
    loader = UniversalDataLoader()
    df = loader.load_daily_equities(tickers=['AAPL'], start_date='2017-01-03', end_date='2017-12-29', source='wiki')
    data = df[['timestamp', 'close']].copy()
    data.set_index('timestamp', inplace=True)

    # Expected signals (from diagnostic script)
    signals = {
        '2017-04-24': ('DEATH', None),  # No position to exit
        '2017-04-26': ('GOLDEN', None),  # Entry
        '2017-06-13': ('DEATH', None),   # Exit
        '2017-07-19': ('GOLDEN', None),  # Entry
        '2017-09-19': ('DEATH', None),   # Exit
        '2017-10-18': ('GOLDEN', None),  # Entry
        '2017-12-11': ('DEATH', None),   # Exit
        '2017-12-20': ('GOLDEN', None),  # Entry
    }

    # Simulate both approaches
    print("\n" + "=" * 80)
    print("VECTORBT APPROACH: Force-close at end")
    print("=" * 80)

    capital = 10000
    position = 0
    cash = capital
    trades = []

    for date_str in sorted(signals.keys()):
        date = pd.to_datetime(date_str)
        signal_type, _ = signals[date_str]
        price = data.loc[date, 'close']

        if signal_type == 'GOLDEN' and position == 0:
            # Buy with all cash
            position = cash / price
            entry_price = price
            entry_date = date
            cash = 0
            print(f"{date.date()}: BUY {position:.2f} shares @ ${price:.2f}")

        elif signal_type == 'DEATH' and position > 0:
            # Sell all shares
            proceeds = position * price
            pnl = proceeds - (position * entry_price)
            cash = proceeds
            print(f"{date.date()}: SELL {position:.2f} shares @ ${price:.2f}, PnL=${pnl:.2f}")
            trades.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': date,
                'exit_price': price,
                'shares': position,
                'pnl': pnl
            })
            position = 0

    # Force close at end if position open
    if position > 0:
        end_date = data.index[-1]
        end_price = data.loc[end_date, 'close']
        proceeds = position * end_price
        pnl = proceeds - (position * entry_price)
        cash = proceeds
        print(f"{end_date.date()}: FORCE-CLOSE {position:.2f} shares @ ${end_price:.2f}, PnL=${pnl:.2f}")
        trades.append({
            'entry_date': entry_price,
            'entry_price': entry_price,
            'exit_date': end_date,
            'exit_price': end_price,
            'shares': position,
            'pnl': pnl
        })
        position = 0

    vbt_final = cash
    vbt_return = (vbt_final / capital - 1) * 100
    vbt_trades = len(trades)

    print(f"\nFinal Value: ${vbt_final:.2f}")
    print(f"Total Return: {vbt_return:.2f}%")
    print(f"Num Trades (round trips): {vbt_trades}")

    # Zipline approach
    print("\n" + "=" * 80)
    print("ZIPLINE APPROACH: Keep position open at end")
    print("=" * 80)

    capital = 10000
    position = 0
    cash = capital
    trades_zl = []

    for date_str in sorted(signals.keys()):
        date = pd.to_datetime(date_str)
        signal_type, _ = signals[date_str]
        price = data.loc[date, 'close']

        if signal_type == 'GOLDEN' and position == 0:
            # Buy with all cash
            position = cash / price
            entry_price = price
            entry_date = date
            cash = 0
            print(f"{date.date()}: BUY {position:.2f} shares @ ${price:.2f}")

        elif signal_type == 'DEATH' and position > 0:
            # Sell all shares
            proceeds = position * price
            pnl = proceeds - (position * entry_price)
            cash = proceeds
            print(f"{date.date()}: SELL {position:.2f} shares @ ${price:.2f}, PnL=${pnl:.2f}")
            trades_zl.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': date,
                'exit_price': price,
                'shares': position,
                'pnl': pnl
            })
            position = 0

    # Calculate unrealized PnL for open position
    if position > 0:
        end_date = data.index[-1]
        end_price = data.loc[end_date, 'close']
        unrealized_pnl = (end_price - entry_price) * position
        print(f"\nOPEN POSITION: {position:.2f} shares, unrealized PnL=${unrealized_pnl:.2f}")

    zl_final = cash + (position * data.iloc[-1]['close'])
    zl_return = (zl_final / capital - 1) * 100
    zl_trades = len(trades_zl)

    print(f"\nFinal Value: ${zl_final:.2f} (${cash:.2f} cash + ${position * data.iloc[-1]['close']:.2f} position)")
    print(f"Total Return: {zl_return:.2f}%")
    print(f"Num Closed Trades: {zl_trades}")

    # Compare
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print(f"\nReturn Difference: {abs(vbt_return - zl_return):.2f}%")
    print(f"Trade Count Difference: {abs(vbt_trades - zl_trades)}")

    print("\nKEY INSIGHT:")
    if vbt_trades > zl_trades:
        print("  VectorBT force-closes final position, Zipline keeps it open")
        print("  This explains the trade count and return difference")
    else:
        print("  Both should have same number of closed trades")

if __name__ == '__main__':
    main()
