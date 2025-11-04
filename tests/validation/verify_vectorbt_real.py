"""
Verify VectorBT multi-asset handling - the real test
This exposes whether we're correctly aggregating multi-asset portfolios
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

print("VectorBT version:", vbt.__version__)
print()

# Create realistic test data
dates = pd.date_range("2020-01-01", "2020-01-31", freq="D")
n_assets = 5

# Create price data with different patterns
np.random.seed(42)
prices = pd.DataFrame(
    {f"Stock{i}": 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01) for i in range(n_assets)},
    index=dates,
)

print("Test Setup:")
print(f"- {n_assets} assets")
print(f"- {len(dates)} days")
print("- Initial capital: $10,000")
print()

# Create alternating entry/exit signals
entries = pd.DataFrame(False, index=dates, columns=prices.columns)
exits = pd.DataFrame(False, index=dates, columns=prices.columns)

# Different entry/exit for each stock to create many trades
for i, col in enumerate(prices.columns):
    # Each stock enters/exits at different times
    entries.loc[dates[5 + i], col] = True
    exits.loc[dates[15 + i], col] = True
    if 20 + i < len(dates):
        entries.loc[dates[20 + i], col] = True
    if 25 + i < len(dates):
        exits.loc[dates[25 + i], col] = True

total_expected_trades = entries.sum().sum() + exits.sum().sum()
print(f"Expected trades: {total_expected_trades}")
print()

# Test 1: Default VectorBT behavior (what we might be doing wrong)
print("=" * 60)
print("TEST 1: Default VectorBT (Separate Portfolios)")
print("=" * 60)

pf1 = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    freq="D",
)

# This returns MULTIPLE values, not a single portfolio!
final_values = pf1.final_value()
print(f"Final values type: {type(final_values)}")
print(f"Final values:\n{final_values}")

if isinstance(final_values, pd.Series):
    print("\n⚠️ WARNING: Getting separate portfolio for each asset!")
    print(f"Sum of final values: ${final_values.sum():.2f}")
    print(f"Mean of final values: ${final_values.mean():.2f}")

# Check trades
print(f"\nNumber of trade records: {len(pf1.orders.records)}")

# Test 2: Grouped portfolio (correct for multi-asset)
print("\n" + "=" * 60)
print("TEST 2: Grouped Portfolio (Single Combined Portfolio)")
print("=" * 60)

pf2 = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    freq="D",
    group_by=True,  # KEY: Group into single portfolio
    cash_sharing=True,  # KEY: Share cash across assets
)

final_value2 = pf2.final_value()
print(f"Final value type: {type(final_value2)}")
print(f"Final value: ${float(final_value2):.2f}")
print(f"Total return: {float(pf2.total_return()) * 100:.2f}%")
print(f"Number of trades: {len(pf2.orders.records)}")

# Test 3: The way QEngine does it (manual calculation)
print("\n" + "=" * 60)
print("TEST 3: QEngine-style Manual Calculation")
print("=" * 60)

cash = 10000
positions = dict.fromkeys(prices.columns, 0)
trades = []

for date in prices.index:
    current_prices = prices.loc[date]

    # Process exits first
    for col in prices.columns:
        if exits.loc[date, col] and positions[col] > 0:
            cash += positions[col] * current_prices[col]
            trades.append(("SELL", col, positions[col], current_prices[col]))
            positions[col] = 0

    # Process entries
    for col in prices.columns:
        if entries.loc[date, col] and positions[col] == 0:
            # Buy $2000 worth of each stock (equal weight)
            shares = 2000 / current_prices[col]
            if cash >= 2000:
                positions[col] = shares
                cash -= shares * current_prices[col]
                trades.append(("BUY", col, shares, current_prices[col]))

# Final portfolio value
final_value_qengine = cash
for col, shares in positions.items():
    if shares > 0:
        final_value_qengine += shares * prices.iloc[-1][col]

print(f"Final value: ${final_value_qengine:.2f}")
print(f"Return: {(final_value_qengine / 10000 - 1) * 100:.2f}%")
print(f"Number of trades: {len(trades)}")

# Compare results
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"Default VectorBT (wrong): ${final_values.mean():.2f} (mean of separate portfolios)")
print(f"Grouped VectorBT (correct): ${float(final_value2):.2f}")
print(f"QEngine-style manual: ${final_value_qengine:.2f}")

diff = abs(float(final_value2) - final_value_qengine)
print(f"\nDifference between grouped VectorBT and QEngine: ${diff:.2f}")

if diff < 0.01:
    print("✅ Results match! VectorBT grouped mode works correctly.")
else:
    print("❌ Results don't match! There's a discrepancy.")

# The smoking gun: Check if we're accidentally comparing wrong values
print("\n" + "=" * 60)
print("THE SMOKING GUN")
print("=" * 60)

# What happens if we mistakenly use the first portfolio's value?
if isinstance(final_values, pd.Series):
    first_asset_value = final_values.iloc[0]
    print(f"First asset portfolio value: ${first_asset_value:.2f}")
    print("This would be WRONG for multi-asset comparison!")

# What if we mistakenly sum all portfolios?
if isinstance(final_values, pd.Series):
    summed_value = final_values.sum()
    print(f"Sum of all separate portfolios: ${summed_value:.2f}")
    print(f"This would be WRONG - it's {n_assets}x the initial capital!")

print("\n⚠️ CRITICAL: For multi-asset portfolios, must use:")
print("  - group_by=True")
print("  - cash_sharing=True")
print("  Otherwise, VectorBT creates SEPARATE $10k portfolios for each asset!")
