# Strategy Development

Learn how to build effective trading strategies with ML4T Backtest.

## Strategy Base Class

All strategies inherit from `Strategy`:

```python
from ml4t.backtest import Strategy

class MyStrategy(Strategy):
    def on_bar(self, bar):
        # Called for each price bar
        pass

    def on_fill(self, fill):
        # Called when an order is filled
        pass
```

## Available Methods

### Order Methods

```python
self.buy(size, price=None)     # Submit buy order
self.sell(size, price=None)    # Submit sell order
self.close()                    # Close current position
self.set_stop(price)           # Set stop loss
self.set_target(price)         # Set profit target
```

### Position Info

```python
self.position      # Current position size (int)
self.equity        # Current equity (float)
self.cash          # Available cash (float)
```

## Strategy Patterns

### Momentum Strategy

```python
class MomentumStrategy(Strategy):
    def __init__(self, fast=10, slow=30):
        self.fast = fast
        self.slow = slow

    def on_bar(self, bar):
        fast_ma = bar.close_ma(self.fast)
        slow_ma = bar.close_ma(self.slow)

        if fast_ma > slow_ma and self.position == 0:
            self.buy(size=100)
        elif fast_ma < slow_ma and self.position > 0:
            self.close()
```

### Risk-Managed Position Sizing

```python
class RiskManagedStrategy(Strategy):
    def __init__(self, risk_per_trade=0.02):
        self.risk_per_trade = risk_per_trade

    def on_bar(self, bar):
        if self.position == 0:
            # Size based on 2% risk
            stop_distance = bar.atr(14) * 2
            size = int(self.equity * self.risk_per_trade / stop_distance)
            self.buy(size=size)
            self.set_stop(bar.close - stop_distance)
```

### Signal-Based Strategy

Use ML model predictions as signals:

```python
class SignalStrategy(Strategy):
    def __init__(self, signals):
        self.signals = signals

    def on_bar(self, bar):
        signal = self.signals.get(bar.datetime)

        if signal > 0.5 and self.position == 0:
            self.buy(size=100)
        elif signal < -0.5 and self.position > 0:
            self.close()
```

## Best Practices

1. **Avoid look-ahead bias**: Only use data available at `bar.datetime`
2. **Account for costs**: Test with realistic commission and slippage
3. **Validate with VectorBT**: Ensure results match expected behavior
4. **Size positions appropriately**: Don't risk more than 1-2% per trade
