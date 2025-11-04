#!/usr/bin/env python3
"""Diagnostic script to understand cross-platform differences."""
import polars as pl
from data import load_test_data
from signals import MACrossoverSignals

# Load data
print("Loading data...")
data = load_test_data(
    'daily_equities',
    start_date='2020-01-01',
    end_date='2020-12-31',
    symbols=['AAPL']
)
print(f"Loaded {len(data)} bars")
print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")

# Generate signals
print("\nGenerating MA crossover signals...")
generator = MACrossoverSignals(fast_period=10, slow_period=30, quantity=100)
signals = generator.generate_signals(data)

print(f"\nGenerated {len(signals)} signals:")
print("-" * 80)
for i, sig in enumerate(signals):
    print(f"{i+1}. {sig.timestamp.date()} - {sig.action:5s} {sig.quantity:3.0f} {sig.symbol}")

    # Find the data row for this signal
    matching_rows = data.filter(pl.col('timestamp') == sig.timestamp)
    if len(matching_rows) > 0:
        row = matching_rows[0]
        print(f"   Signal date data: O={row['open'][0]:.2f} H={row['high'][0]:.2f} L={row['low'][0]:.2f} C={row['close'][0]:.2f}")
    else:
        print(f"   ⚠️  No matching data row found!")

    # Find next bar (for QEngine-style next-bar execution)
    next_rows = data.filter(pl.col('timestamp') > sig.timestamp).sort('timestamp').head(1)
    if len(next_rows) > 0:
        next_row = next_rows[0]
        print(f"   Next bar ({next_row['timestamp'][0].date()}): O={next_row['open'][0]:.2f} H={next_row['high'][0]:.2f} L={next_row['low'][0]:.2f} C={next_row['close'][0]:.2f}")

    print()

print("\nExpected execution patterns:")
print("-" * 80)
print("QEngine: Executes at next market event after signal")
print("VectorBT: Executes on the signal bar (same-bar execution)")
print("\nThis explains the 1-day difference in entry times.")
