#!/usr/bin/env python3
"""VectorBT Pro Hello World - Simple MA Crossover Backtest"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np

print(f"✅ VectorBT Pro version: {vbt.__version__}")

# Create simple price data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
price = pd.Series(
    100 + np.cumsum(np.random.randn(100) * 2),
    index=dates,
    name='price'
)

print(f"\n📊 Price data: {len(price)} days")
print(f"   Start: ${price.iloc[0]:.2f}")
print(f"   End: ${price.iloc[-1]:.2f}")

# Simple MA crossover strategy
fast_ma = vbt.MA.run(price, window=10)
slow_ma = vbt.MA.run(price, window=30)

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

print(f"\n📈 Strategy signals:")
print(f"   Buy signals: {entries.sum()}")
print(f"   Sell signals: {exits.sum()}")

# Run backtest
portfolio = vbt.Portfolio.from_signals(
    price,
    entries,
    exits,
    init_cash=10000,
    fees=0.001  # 0.1% commission
)

print(f"\n💰 Portfolio Results:")
print(f"   Initial cash: ${portfolio.init_cash:.2f}")
print(f"   Final value: ${portfolio.final_value:.2f}")
print(f"   Total return: {portfolio.total_return:.2%}")
print(f"   Total trades: {portfolio.trades.count()}")
print(f"   Win rate: {portfolio.trades.win_rate:.2%}")
print(f"   Sharpe ratio: {portfolio.sharpe_ratio:.2f}")

print(f"\n🎉 VectorBT Pro installation verified successfully!")
