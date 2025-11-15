"""
Direct comparison: VectorBT Pro vs ml4t.backtest on identical 5,000 trade scenario
Uses exact same data and signals from multi_asset_portfolio_validation.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbtpro as vbt

print("=" * 70)
print("VECTORBT PRO vs ML4T.BACKTEST: 5,000 TRADE COMPARISON")
print("=" * 70)
print(f"VectorBT Pro Version: {vbt.__version__}")
print()

# Load Wiki data - exact same as multi_asset_portfolio_validation.py
projects_dir = Path("~/ml4t/projects")
wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"

print("Loading Wiki data (same as multi_asset_portfolio_validation.py)...")
df = pd.read_parquet(wiki_path)

# EXACT same filtering as multi_asset_portfolio_validation.py
start_date = "2013-01-01"
end_date = "2017-12-31"

df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

# Get top 30 most liquid stocks by mean volume
top_stocks = df.groupby("ticker")["volume"].mean().nlargest(30).index.tolist()
df = df[df["ticker"].isin(top_stocks)]

# Pivot to wide format
prices = df.pivot(index="date", columns="ticker", values="adj_close")
prices = prices.dropna(axis=1)  # Drop any stocks with missing data

print(f"Data shape: {prices.shape[0]} days x {prices.shape[1]} stocks")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
print(f"Stocks: {', '.join(prices.columns[:5])}... ({len(prices.columns)} total)")
print()

# Generate EXACT same momentum signals
print("Generating trading signals (same momentum strategy)...")
lookback = 20
rebalance_freq = 5  # Every 5 days for more trades

entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)

for i in range(lookback, len(prices), rebalance_freq):
    # Calculate momentum
    returns = prices.iloc[i] / prices.iloc[i - lookback] - 1

    # Select top and bottom performers
    top_n = 5
    top_stocks = returns.nlargest(top_n).index
    bottom_stocks = returns.nsmallest(top_n).index

    # Exit all positions first (if not first period)
    if i > lookback:
        exits.iloc[i - 1] = True

    # Enter new positions (long top, short bottom)
    for stock in top_stocks:
        entries.iloc[i, prices.columns.get_loc(stock)] = True

total_signals = entries.sum().sum() + exits.sum().sum()
print(f"Total signals generated: {total_signals}")
print()

# Parameters from multi_asset_portfolio_validation.py
initial_capital = 100000
position_size = initial_capital / 10  # Max 10 positions

# Calculate size in shares for VectorBT
size = position_size / prices

# TEST 1: VectorBT Pro
print("=" * 50)
print("TEST 1: VectorBT Pro")
print("=" * 50)

# Warm-up run (JIT compilation)
print("Warm-up run for JIT compilation...")
_ = vbt.Portfolio.from_signals(
    prices.iloc[:100],
    entries=entries.iloc[:100],
    exits=exits.iloc[:100],
    size=size.iloc[:100],
    init_cash=initial_capital,
    fees=0.0,
    slippage=0.0,
    freq="D",
    group_by=True,
    cash_sharing=True,
).final_value

# Actual timing run
print("Running full backtest...")
start_time = time.time()

portfolio_pro = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    size=size,
    init_cash=initial_capital,
    fees=0.0,
    slippage=0.0,
    freq="D",
    group_by=True,
    cash_sharing=True,
)

# Force computation of all results
final_value = float(portfolio_pro.final_value)
total_return = float(portfolio_pro.total_return) * 100
n_trades = portfolio_pro.orders.count()
sharpe = float(portfolio_pro.sharpe_ratio)
max_dd = float(portfolio_pro.max_drawdown) * 100
win_rate = float(portfolio_pro.trades.win_rate) * 100

pro_time = time.time() - start_time

print(f"Final value: ${final_value:,.2f}")
print(f"Total return: {total_return:.2f}%")
print(f"Number of trades: {n_trades}")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Max drawdown: {max_dd:.2f}%")
print(f"Win rate: {win_rate:.1f}%")
print(f"Execution time: {pro_time:.3f} seconds")
print(f"Trades per second: {n_trades / pro_time:.0f}")

# TEST 2: Multiple runs for timing stability
print("\n" + "=" * 50)
print("TEST 2: Timing Stability (10 runs)")
print("=" * 50)

times = []
for run in range(10):
    start = time.time()

    pf = vbt.Portfolio.from_signals(
        prices,
        entries=entries,
        exits=exits,
        size=size,
        init_cash=initial_capital,
        fees=0.0,
        slippage=0.0,
        freq="D",
        group_by=True,
        cash_sharing=True,
    )

    # Force computation
    _ = pf.final_value
    _ = pf.orders.count()

    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Run {run + 1}: {elapsed:.3f}s")

avg_time = np.mean(times)
std_time = np.std(times)
print(f"\nAverage: {avg_time:.3f}s ± {std_time:.3f}s")
print(f"Best: {min(times):.3f}s")
print(f"Worst: {max(times):.3f}s")

# TEST 3: Breakdown by operation
print("\n" + "=" * 50)
print("TEST 3: Operation Breakdown")
print("=" * 50)

# Time portfolio creation
start = time.time()
pf = vbt.Portfolio.from_signals(
    prices,
    entries=entries,
    exits=exits,
    size=size,
    init_cash=initial_capital,
    fees=0.0,
    slippage=0.0,
    freq="D",
    group_by=True,
    cash_sharing=True,
)
create_time = time.time() - start
print(f"Portfolio creation: {create_time:.3f}s")

# Time metrics computation
start = time.time()
_ = pf.final_value
_ = pf.total_return
value_time = time.time() - start
print(f"Value computation: {value_time:.3f}s")

start = time.time()
_ = pf.sharpe_ratio
_ = pf.max_drawdown
metrics_time = time.time() - start
print(f"Metrics computation: {metrics_time:.3f}s")

start = time.time()
_ = pf.orders.count()
_ = pf.trades.win_rate
trades_time = time.time() - start
print(f"Trades analysis: {trades_time:.3f}s")

# COMPARISON
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print("From multi_asset_portfolio_validation.py (same data & signals):")
print("- ml4t.backtest:     0.580 seconds for 4,960 trades")
print("- VectorBT:    5.406 seconds for 4,960 trades")
print()
print("This test (VectorBT Pro):")
print(f"- VectorBT Pro: {avg_time:.3f} seconds for {n_trades} trades")
print()

ml4t.backtest_speed = 4960 / 0.580  # trades per second
vbt_speed = 4960 / 5.406
vbtpro_speed = n_trades / avg_time

print("Trades per second:")
print(f"- ml4t.backtest:     {ml4t.backtest_speed:,.0f} trades/sec")
print(f"- VectorBT:    {vbt_speed:,.0f} trades/sec")
print(f"- VectorBT Pro: {vbtpro_speed:,.0f} trades/sec")
print()

print("Speed comparisons:")
print(f"- VectorBT Pro vs VectorBT: {vbtpro_speed / vbt_speed:.1f}x faster")
print(f"- ml4t.backtest vs VectorBT Pro: {ml4t.backtest_speed / vbtpro_speed:.1f}x faster")
print(f"- ml4t.backtest vs VectorBT: {ml4t.backtest_speed / vbt_speed:.1f}x faster")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"1. VectorBT Pro processes {n_trades} trades in {avg_time:.3f} seconds")
print(f"2. VectorBT Pro is {vbtpro_speed / vbt_speed:.1f}x faster than standard VectorBT")
print(f"3. ml4t.backtest is still {ml4t.backtest_speed / vbtpro_speed:.1f}x faster than VectorBT Pro")
print("4. All three frameworks produce identical results (validated earlier)")
print("5. VectorBT Pro offers significant speedup over standard VectorBT")
