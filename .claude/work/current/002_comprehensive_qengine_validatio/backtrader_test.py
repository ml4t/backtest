#!/usr/bin/env python3
"""Backtrader Hello World - Simple MA Crossover Strategy"""

import backtrader as bt
import pandas as pd
import datetime

print(f"✅ Backtrader version: {bt.__version__}")

class SimpleMAStrategy(bt.Strategy):
    params = (('fast', 10), ('slow', 30),)

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow
        )
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:  # Fast MA crosses above slow MA
                self.buy()
        else:
            if self.crossover < 0:  # Fast MA crosses below slow MA
                self.close()

# Create synthetic price data for testing
dates = pd.date_range('2020-01-01', periods=100, freq='D')
import numpy as np
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(100) * 2)

data_df = pd.DataFrame({
    'open': prices,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'close': prices,
    'volume': 100000,
}, index=dates)

print(f"\n📊 Price data: {len(data_df)} days")
print(f"   Start: ${data_df['close'].iloc[0]:.2f}")
print(f"   End: ${data_df['close'].iloc[-1]:.2f}")

# Create Cerebro engine
cerebro = bt.Cerebro()
cerebro.addstrategy(SimpleMAStrategy)

# Add data
data = bt.feeds.PandasData(dataname=data_df)
cerebro.adddata(data)

# Set initial cash
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

print(f"\n💰 Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")

# Run backtest
cerebro.run()

print(f"💰 Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
total_return = (cerebro.broker.getvalue() / 10000 - 1) * 100
print(f"   Total return: {total_return:.2f}%")

print(f"\n🎉 Backtrader installation verified successfully!")
