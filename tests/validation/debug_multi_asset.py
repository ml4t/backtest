"""
Debug why multi-asset portfolios appear to match perfectly
"""

import pandas as pd
import vectorbt as vbt

# Simple 3-asset test
dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
n_assets = 3

# Different price movements
prices = pd.DataFrame(
    {
        "A": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
        "B": [100, 99, 98, 97, 98, 99, 100, 101, 102, 103],
        "C": [100, 100, 101, 101, 102, 102, 101, 101, 100, 100],
    },
    index=dates,
)

print("Prices:")
print(prices)
print()

# Simple signals - each asset trades once
entries = pd.DataFrame(False, index=dates, columns=prices.columns)
exits = pd.DataFrame(False, index=dates, columns=prices.columns)

entries.iloc[2, 0] = True  # Buy A on day 2
entries.iloc[3, 1] = True  # Buy B on day 3
entries.iloc[4, 2] = True  # Buy C on day 4

exits.iloc[7, 0] = True  # Sell A on day 7
exits.iloc[8, 1] = True  # Sell B on day 8
exits.iloc[9, 2] = True  # Sell C on day 9

print("Total expected trades:", entries.sum().sum() + exits.sum().sum())

# Method 1: QEngine style (manual)
print("\n" + "=" * 50)
print("Method 1: QEngine Manual")
print("=" * 50)

cash = 10000
positions = {"A": 0, "B": 0, "C": 0}
position_size = 3333  # Fixed size per position

for i, date in enumerate(dates):
    # Process exits
    for col in prices.columns:
        if exits.loc[date, col] and positions[col] > 0:
            cash += positions[col] * prices.loc[date, col]
            print(
                f"Day {i}: SELL {col} - {positions[col]:.2f} shares @ ${prices.loc[date, col]:.2f}",
            )
            positions[col] = 0

    # Process entries
    for col in prices.columns:
        if entries.loc[date, col] and positions[col] == 0:
            shares = position_size / prices.loc[date, col]
            if cash >= position_size:
                positions[col] = shares
                cash -= shares * prices.loc[date, col]
                print(f"Day {i}: BUY {col} - {shares:.2f} shares @ ${prices.loc[date, col]:.2f}")

# Final value
final_qengine = cash
for col, shares in positions.items():
    if shares > 0:
        final_qengine += shares * prices.iloc[-1][col]

print(f"\nFinal cash: ${cash:.2f}")
print(f"Final portfolio value: ${final_qengine:.2f}")

# Method 2: VectorBT WITHOUT grouping
print("\n" + "=" * 50)
print("Method 2: VectorBT WITHOUT Grouping")
print("=" * 50)

pf_separate = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
)

final_separate = pf_separate.final_value()
print(f"Type: {type(final_separate)}")
print(f"Values: {final_separate}")
if isinstance(final_separate, pd.Series):
    print(f"Mean: ${final_separate.mean():.2f}")
    print("THIS IS WRONG for portfolio comparison!")

# Method 3: VectorBT WITH grouping and cash sharing
print("\n" + "=" * 50)
print("Method 3: VectorBT WITH Grouping")
print("=" * 50)

pf_grouped = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    group_by=True,
    cash_sharing=True,
)

final_grouped = float(pf_grouped.final_value())
print(f"Final value: ${final_grouped:.2f}")
print(f"Trades executed: {len(pf_grouped.orders.records)}")

# Method 4: VectorBT with explicit position sizing
print("\n" + "=" * 50)
print("Method 4: VectorBT WITH Sizing")
print("=" * 50)

# Position size (in shares)
size = position_size / prices

pf_sized = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    size=size,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    group_by=True,
    cash_sharing=True,
)

final_sized = float(pf_sized.final_value())
print(f"Final value: ${final_sized:.2f}")
print(f"Trades executed: {len(pf_sized.orders.records)}")

# Show the trades
if len(pf_sized.orders.records) > 0:
    print("\nTrades:")
    for r in pf_sized.orders.records:
        print(f"  Col {r['col']}: Size={r['size']:.2f}, Price=${r['price']:.2f}")

# COMPARISON
print("\n" + "=" * 50)
print("RESULTS COMPARISON")
print("=" * 50)

print(f"QEngine manual: ${final_qengine:.2f}")
print(f"VectorBT separate (WRONG): ${final_separate.mean():.2f}")
print(f"VectorBT grouped: ${final_grouped:.2f}")
print(f"VectorBT sized: ${final_sized:.2f}")

print("\n🚨 THE ISSUE:")
if abs(final_qengine - final_grouped) > 0.01:
    print(f"Grouped VectorBT differs from QEngine by ${abs(final_qengine - final_grouped):.2f}")
    print("This suggests VectorBT's cash_sharing might work differently than expected!")

if abs(final_qengine - final_sized) < 0.01:
    print("✅ VectorBT with explicit sizing matches QEngine")
else:
    print(f"❌ Even with sizing, VectorBT differs by ${abs(final_qengine - final_sized):.2f}")
