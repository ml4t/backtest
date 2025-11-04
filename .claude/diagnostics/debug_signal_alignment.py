#!/usr/bin/env python3
"""
Debug script to compare MA calculations between Zipline and VectorBT approaches.
Extracts exact MA values on critical signal dates to identify discrepancies.
"""

import pandas as pd
import numpy as np
from tests.validation.data_loader import UniversalDataLoader

# Critical dates from reconciliation doc
CRITICAL_DATES = [
    '2017-04-24',  # Death cross
    '2017-04-25',  # Zipline entry (golden cross?)
    '2017-04-26',  # VectorBT entry (golden cross confirmed)
    '2017-06-12',  # Day before VectorBT exit
    '2017-06-13',  # VectorBT exit (death cross)
    '2017-09-19',  # Zipline exit (death cross?)
]

def calculate_mas_vectorbt_style(data, short_window=10, long_window=30):
    """Calculate MAs using VectorBT's approach (pandas rolling)."""
    close = data['close']

    ma_short = close.rolling(window=short_window).mean()
    ma_long = close.rolling(window=long_window).mean()

    # Crossover detection (VectorBT style)
    golden_cross = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
    death_cross = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

    return pd.DataFrame({
        'close': close,
        'ma_short': ma_short,
        'ma_long': ma_long,
        'prev_ma_short': ma_short.shift(1),
        'prev_ma_long': ma_long.shift(1),
        'golden_cross': golden_cross,
        'death_cross': death_cross
    })

def calculate_mas_zipline_style(data, short_window=10, long_window=30):
    """Calculate MAs using Zipline's current approach (history slicing)."""
    close = data['close'].values
    results = []

    for i in range(len(data)):
        if i < long_window:
            results.append({
                'date': data.index[i],
                'close': close[i],
                'ma_short': np.nan,
                'ma_long': np.nan,
                'prev_ma_short': np.nan,
                'prev_ma_long': np.nan,
                'golden_cross': False,
                'death_cross': False
            })
            continue

        # Current window (includes today) - get long_window + 1 for prev calculation
        history = close[max(0, i - long_window):i + 1]
        ma_short = history[-short_window:].mean()
        ma_long = history[-long_window:].mean()  # FIX: Only use last long_window days

        # Previous window (excludes today)
        prev_history = history[:-1]
        prev_ma_short = prev_history[-short_window:].mean()
        prev_ma_long = prev_history[-long_window:].mean()  # FIX: Only use last long_window days

        # Crossover detection
        golden_cross = (prev_ma_short <= prev_ma_long) and (ma_short > ma_long)
        death_cross = (prev_ma_short > prev_ma_long) and (ma_short <= ma_long)

        results.append({
            'date': data.index[i],
            'close': close[i],
            'ma_short': ma_short,
            'ma_long': ma_long,
            'prev_ma_short': prev_ma_short,
            'prev_ma_long': prev_ma_long,
            'golden_cross': golden_cross,
            'death_cross': death_cross
        })

    df = pd.DataFrame(results)
    df.set_index('date', inplace=True)
    return df

def main():
    print("=" * 80)
    print("ZIPLINE vs VECTORBT MA CALCULATION COMPARISON")
    print("=" * 80)

    # Load data
    loader = UniversalDataLoader()
    df = loader.load_daily_equities(tickers=['AAPL'], start_date='2017-01-03', end_date='2017-12-29', source='wiki')

    # Convert to single-ticker time series (needed for analysis)
    data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    data.set_index('timestamp', inplace=True)
    data.index = pd.to_datetime(data.index)

    print(f"\nLoaded {len(data)} days of AAPL data")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

    # Calculate both ways
    print("\n" + "=" * 80)
    print("VECTORBT APPROACH (pandas rolling)")
    print("=" * 80)
    vbt_mas = calculate_mas_vectorbt_style(data)

    print("\n" + "=" * 80)
    print("ZIPLINE APPROACH (history slicing)")
    print("=" * 80)
    zipline_mas = calculate_mas_zipline_style(data)

    # Compare on critical dates
    print("\n" + "=" * 80)
    print("COMPARISON ON CRITICAL DATES")
    print("=" * 80)

    for date_str in CRITICAL_DATES:
        date = pd.to_datetime(date_str)
        if date not in vbt_mas.index or date not in zipline_mas.index:
            print(f"\n{date_str}: DATE NOT IN DATA")
            continue

        vbt = vbt_mas.loc[date]
        zl = zipline_mas.loc[date]

        print(f"\n{date_str}:")
        print(f"  Close: ${vbt['close']:.2f}")
        print(f"\n  VectorBT:")
        print(f"    MA(10) curr: {vbt['ma_short']:.4f}, prev: {vbt['prev_ma_short']:.4f}")
        print(f"    MA(30) curr: {vbt['ma_long']:.4f}, prev: {vbt['prev_ma_long']:.4f}")
        print(f"    Golden: {vbt['golden_cross']}, Death: {vbt['death_cross']}")

        print(f"\n  Zipline:")
        print(f"    MA(10) curr: {zl['ma_short']:.4f}, prev: {zl['prev_ma_short']:.4f}")
        print(f"    MA(30) curr: {zl['ma_long']:.4f}, prev: {zl['prev_ma_long']:.4f}")
        print(f"    Golden: {zl['golden_cross']}, Death: {zl['death_cross']}")

        # Check for differences
        ma_diff = abs(vbt['ma_short'] - zl['ma_short'])
        if ma_diff > 0.01:
            print(f"\n  ⚠️  MA(10) DIFFERENCE: {ma_diff:.4f}")

        if vbt['golden_cross'] != zl['golden_cross']:
            print(f"\n  ❌ GOLDEN CROSS MISMATCH!")

        if vbt['death_cross'] != zl['death_cross']:
            print(f"\n  ❌ DEATH CROSS MISMATCH!")

    # Find all signals
    print("\n" + "=" * 80)
    print("ALL SIGNALS DETECTED")
    print("=" * 80)

    vbt_golden = vbt_mas[vbt_mas['golden_cross']].index.tolist()
    vbt_death = vbt_mas[vbt_mas['death_cross']].index.tolist()
    zl_golden = zipline_mas[zipline_mas['golden_cross']].index.tolist()
    zl_death = zipline_mas[zipline_mas['death_cross']].index.tolist()

    print(f"\nVectorBT Golden Crosses ({len(vbt_golden)}):")
    for date in vbt_golden:
        print(f"  {date.date()}")

    print(f"\nVectorBT Death Crosses ({len(vbt_death)}):")
    for date in vbt_death:
        print(f"  {date.date()}")

    print(f"\nZipline Golden Crosses ({len(zl_golden)}):")
    for date in zl_golden:
        print(f"  {date.date()}")

    print(f"\nZipline Death Crosses ({len(zl_death)}):")
    for date in zl_death:
        print(f"  {date.date()}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"VectorBT: {len(vbt_golden)} golden, {len(vbt_death)} death crosses")
    print(f"Zipline:  {len(zl_golden)} golden, {len(zl_death)} death crosses")

    if vbt_golden == zl_golden and vbt_death == zl_death:
        print("\n✅ SIGNALS MATCH PERFECTLY!")
    else:
        print("\n❌ SIGNALS DO NOT MATCH - INVESTIGATION NEEDED")

if __name__ == '__main__':
    main()
