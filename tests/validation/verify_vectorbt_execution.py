"""
Verify VectorBT is actually executing trades properly
Let's test with a simple example to ensure we're not cheating
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

# Create simple test data
dates = pd.date_range("2020-01-01", "2020-01-31", freq="D")
n_assets = 3
n_days = len(dates)

# Create price data - different trends for each asset
prices = pd.DataFrame(
    {
        "Asset1": 100 + np.arange(n_days) * 0.5,  # Uptrend
        "Asset2": 100 - np.arange(n_days) * 0.3,  # Downtrend
        "Asset3": 100 + np.sin(np.arange(n_days) * 0.5) * 5,  # Oscillating
    },
    index=dates,
)

print("Test Data:")
print(prices.head())
print(f"\nShape: {prices.shape}")

# Create simple signals - buy on day 5, sell on day 15 for each asset
entries = pd.DataFrame(False, index=dates, columns=prices.columns)
exits = pd.DataFrame(False, index=dates, columns=prices.columns)

entries.iloc[5] = True  # Buy all on day 5
exits.iloc[15] = True  # Sell all on day 15

print("\nEntry signals:")
print(entries[entries.any(axis=1)])
print("\nExit signals:")
print(exits[exits.any(axis=1)])

# Test 1: Individual portfolios (no cash sharing)
print("\n" + "=" * 60)
print("TEST 1: Individual Portfolios (No Cash Sharing)")
print("=" * 60)

pf_individual = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    freq="D",
)

print(f"Final value: ${pf_individual.final_value()}")
print(f"Total return: {pf_individual.total_return() * 100:.2f}%")
print(f"Number of trades: {len(pf_individual.orders.records)}")

# Show trades
if len(pf_individual.orders.records) > 0:
    print("\nTrades executed:")
    for record in pf_individual.orders.records:
        print(f"  Asset {record['col']}: Size={record['size']:.2f}, Price=${record['price']:.2f}")

# Test 2: Grouped portfolio with cash sharing
print("\n" + "=" * 60)
print("TEST 2: Grouped Portfolio (With Cash Sharing)")
print("=" * 60)

pf_grouped = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    freq="D",
    group_by=True,  # Group all assets
    cash_sharing=True,  # Share cash
)

print(f"Final value: ${pf_grouped.final_value()}")
print(f"Total return: {pf_grouped.total_return() * 100:.2f}%")
print(f"Number of trades: {len(pf_grouped.orders.records)}")

# Show trades
if len(pf_grouped.orders.records) > 0:
    print("\nTrades executed:")
    for record in pf_grouped.orders.records:
        print(f"  Asset {record['col']}: Size={record['size']:.2f}, Price=${record['price']:.2f}")

# Test 3: With position sizing
print("\n" + "=" * 60)
print("TEST 3: With Fixed Position Sizing")
print("=" * 60)

# Fixed size per position
position_size = 1000 / prices  # $1000 per position

pf_sized = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    size=position_size,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    freq="D",
    group_by=True,
    cash_sharing=True,
)

print(f"Final value: ${pf_sized.final_value()}")
print(f"Total return: {pf_sized.total_return() * 100:.2f}%")
print(f"Number of trades: {len(pf_sized.orders.records)}")

# Show trades
if len(pf_sized.orders.records) > 0:
    print("\nTrades executed:")
    for record in pf_sized.orders.records:
        print(f"  Asset {record['col']}: Size={record['size']:.2f}, Price=${record['price']:.2f}")

# Test 4: Verify trade counting
print("\n" + "=" * 60)
print("TEST 4: Trade Counting Verification")
print("=" * 60)

# Create many trades
many_entries = pd.DataFrame(False, index=dates, columns=prices.columns)
many_exits = pd.DataFrame(False, index=dates, columns=prices.columns)

# Alternate buy/sell every 3 days for each asset
for i in range(0, n_days, 6):
    if i < n_days:
        many_entries.iloc[i] = True
    if i + 3 < n_days:
        many_exits.iloc[i + 3] = True

print(f"Expected entries: {many_entries.sum().sum()}")
print(f"Expected exits: {many_exits.sum().sum()}")
print(f"Expected total trades: {many_entries.sum().sum() + many_exits.sum().sum()}")

pf_many = vbt.Portfolio.from_signals(
    prices,
    entries=many_entries,
    exits=many_exits,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    freq="D",
)

print(f"\nActual trades executed: {len(pf_many.orders.records)}")
print(f"Final value: ${pf_many.final_value()}")

# Detailed trade analysis
trade_summary = pd.DataFrame(pf_many.orders.records)
if not trade_summary.empty:
    print("\nTrade summary by asset:")
    print(trade_summary.groupby("col").size())

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("If VectorBT is working correctly:")
print("- Trade counts should match expected")
print("- Different configurations should give different results")
print("- Cash sharing should affect position sizing")
