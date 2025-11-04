"""
EXPOSE THE TRUTH: Is the VectorBT implementation actually correct?
"""

import pandas as pd
import vectorbt as vbt

# Create a simple scenario where differences MUST show
dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")

# Two assets with opposite movements
prices = pd.DataFrame(
    {
        "UP": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],  # Always up
        "DOWN": [100, 95, 90, 85, 80, 75, 70, 65, 60, 55],  # Always down
    },
    index=dates,
)

print("Test scenario: One stock goes UP, one goes DOWN")
print(prices)
print()

# Buy both on day 1, sell both on day 8
entries = pd.DataFrame(False, index=dates, columns=prices.columns)
exits = pd.DataFrame(False, index=dates, columns=prices.columns)

entries.iloc[1] = True  # Buy both on day 1
exits.iloc[8] = True  # Sell both on day 8

initial_capital = 10000

print("=" * 60)
print("QEngine Approach (Equal Dollar Allocation)")
print("=" * 60)

# QEngine: Allocate $5000 to each stock
cash = initial_capital
positions = {}

# Day 1: Buy
for col in prices.columns:
    shares = 5000 / prices.iloc[1][col]  # $5000 worth
    positions[col] = shares
    cash -= shares * prices.iloc[1][col]
    print(
        f"BUY {col}: {shares:.2f} shares @ ${prices.iloc[1][col]:.2f} = ${shares * prices.iloc[1][col]:.2f}",
    )

print(f"Cash remaining: ${cash:.2f}")

# Day 8: Sell
final_value = cash
for col in prices.columns:
    value = positions[col] * prices.iloc[8][col]
    final_value += value
    print(f"SELL {col}: {positions[col]:.2f} shares @ ${prices.iloc[8][col]:.2f} = ${value:.2f}")

print(f"Final value: ${final_value:.2f}")
print(f"Return: {(final_value / initial_capital - 1) * 100:.2f}%")

qengine_final = final_value

print("\n" + "=" * 60)
print("VectorBT Approach 1: With Fixed Dollar Size")
print("=" * 60)

# This is what we claim to be doing
size_dollars = 5000  # $5000 per position
size_shares = size_dollars / prices  # Convert to shares

pf1 = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    size=size_shares,  # Size in SHARES
    init_cash=initial_capital,
    fees=0.0,
    slippage=0.0,
    group_by=True,
    cash_sharing=True,
)

vbt_final_1 = float(pf1.final_value())
print(f"Final value: ${vbt_final_1:.2f}")
print(f"Return: {float(pf1.total_return()) * 100:.2f}%")
print(f"Trades: {len(pf1.orders.records)}")

print("\n" + "=" * 60)
print("VectorBT Approach 2: Default (100% allocation)")
print("=" * 60)

# What if we don't specify size?
pf2 = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    init_cash=initial_capital,
    fees=0.0,
    slippage=0.0,
    group_by=True,
    cash_sharing=True,
)

vbt_final_2 = float(pf2.final_value())
print(f"Final value: ${vbt_final_2:.2f}")
print(f"Return: {float(pf2.total_return()) * 100:.2f}%")
print(f"Trades: {len(pf2.orders.records)}")

print("\n" + "=" * 60)
print("VectorBT Approach 3: The WRONG way (our bug?)")
print("=" * 60)

# What if we calculate size wrong?
size_wrong = initial_capital / 2 / prices  # This would buy too many shares!

pf3 = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    size=size_wrong,
    init_cash=initial_capital,
    fees=0.0,
    slippage=0.0,
    group_by=True,
    cash_sharing=True,
)

vbt_final_3 = float(pf3.final_value())
print(f"Final value: ${vbt_final_3:.2f}")
print(f"Return: {float(pf3.total_return()) * 100:.2f}%")
print(f"Trades: {len(pf3.orders.records)}")

# Check what actually got executed
if len(pf3.orders.records) > 0:
    print("\nActual trades executed:")
    for _i, r in enumerate(pf3.orders.records):
        col_idx = r[0]  # Column index
        size = r[1]  # Size
        price = r[2]  # Price
        col_name = prices.columns[col_idx]
        print(f"  {col_name}: {size:.2f} shares @ ${price:.2f} = ${size * price:.2f}")

print("\n" + "=" * 60)
print("THE TRUTH")
print("=" * 60)

print(f"QEngine result: ${qengine_final:.2f}")
print(f"VectorBT (correct sizing): ${vbt_final_1:.2f}")
print(f"VectorBT (no sizing): ${vbt_final_2:.2f}")
print(f"VectorBT (wrong sizing): ${vbt_final_3:.2f}")

print("\n🔍 Analysis:")
if abs(qengine_final - vbt_final_1) < 0.01:
    print("✅ VectorBT with correct sizing matches QEngine")
else:
    print(f"❌ Even with 'correct' sizing, differs by ${abs(qengine_final - vbt_final_1):.2f}")

if vbt_final_2 != vbt_final_1:
    print("⚠️ Default VectorBT (no sizing) gives different results!")

print("\n💡 The issue:")
print("VectorBT's 'size' parameter is in SHARES, not dollars!")
print("So size = dollars / price is correct")
print("But size must be calculated at the entry point, not as a DataFrame!")
