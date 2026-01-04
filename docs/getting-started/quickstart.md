# Quickstart

Build and run your first backtest in 5 minutes.

## Basic Strategy

```python
import polars as pl
from ml4t.backtest import Engine, Strategy, BacktestConfig

# Create sample data
data = pl.DataFrame({
    "timestamp": pl.date_range(start="2023-01-01", periods=100, eager=True),
    "open": [100 + i * 0.1 for i in range(100)],
    "high": [101 + i * 0.1 for i in range(100)],
    "low": [99 + i * 0.1 for i in range(100)],
    "close": [100.5 + i * 0.1 for i in range(100)],
    "volume": [1000] * 100,
})

# Define strategy
class BuyAndHold(Strategy):
    def on_bar(self, bar):
        if self.position == 0:
            self.buy(size=100)

# Configure and run
config = BacktestConfig(initial_cash=100_000)
engine = Engine(data, BuyAndHold(), config)
result = engine.run()

# View results
print(f"Final Equity: ${result.final_equity:,.2f}")
print(f"Total Return: {result.total_return:.2%}")
```

## Mean Reversion Strategy

```python
import numpy as np

class MeanReversion(Strategy):
    def __init__(self, lookback=20, z_threshold=2.0):
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.prices = []

    def on_bar(self, bar):
        self.prices.append(bar.close)
        if len(self.prices) < self.lookback:
            return

        mean = np.mean(self.prices[-self.lookback:])
        std = np.std(self.prices[-self.lookback:])
        z_score = (bar.close - mean) / std

        if z_score < -self.z_threshold and self.position <= 0:
            self.buy(size=100)
        elif z_score > self.z_threshold and self.position >= 0:
            self.sell(size=100)
```

## Adding Transaction Costs

```python
config = BacktestConfig(
    initial_cash=100_000,
    commission=0.001,    # 0.1% per trade
    slippage=0.0005,     # 0.05% slippage
)
```

## Using Presets

```python
from ml4t.backtest.presets import crypto_futures, us_equities

# Binance-like fees
config = crypto_futures()

# IB-like fees
config = us_equities()
```

## Analyzing Results

```python
result = engine.run()

# Key metrics
print(result.metrics)
# {'sharpe': 1.23, 'max_drawdown': -0.15, 'total_return': 0.45, ...}

# Equity curve
equity_df = result.equity_curve
print(equity_df)

# Trade list
for trade in result.trades:
    print(f"{trade.entry_date} -> {trade.exit_date}: {trade.pnl:.2f}")
```

## Next Steps

- [Strategy Patterns](../user-guide/strategies.md) - Advanced strategies
- [Order Types](../user-guide/orders.md) - Limit, stop, and stop-limit orders
- [Account Policies](../user-guide/accounts.md) - Cash vs margin accounts
