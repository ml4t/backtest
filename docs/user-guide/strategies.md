# Strategy Development

Learn how to build effective trading strategies with ML4T Backtest.

## Strategy Base Class

All strategies inherit from `Strategy` and implement `on_data(...)`:

```python
from ml4t.backtest.strategy import Strategy

class MyStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        # Called once per bar/timestamp.
        # data: {asset: {"open","high","low","close","volume",...}}
        pass
```

Optional lifecycle hooks:

```python
def on_start(self, broker): ...
def on_end(self, broker): ...
```

## Trading Through The Broker

Strategies submit orders via the provided `broker` object.

### Core broker methods

```python
broker.submit_order("AAPL", 100)               # buy market
broker.submit_order("AAPL", -100)              # sell market
broker.submit_order("AAPL", 100, order_type=...)  # limit/stop/trailing stop
broker.close_position("AAPL")
broker.order_target_percent("AAPL", 0.25)
broker.order_target_value("AAPL", 50_000)
broker.get_position("AAPL")
broker.get_account_value()
broker.get_cash()
```

## Strategy Patterns

### Signal threshold (single asset)

```python
class SignalStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        bar = data.get("AAPL")
        if not bar:
            return
        signal = context.get("signal", 0.0)
        pos = broker.get_position("AAPL")
        if signal > 0.5 and pos is None:
            broker.submit_order("AAPL", 100)
        elif signal < -0.5 and pos is not None:
            broker.close_position("AAPL")
```

### Multi-asset rebalance

```python
class RebalanceStrategy(Strategy):
    def __init__(self, target_weights):
        self.target_weights = target_weights

    def on_data(self, timestamp, data, context, broker):
        broker.rebalance_to_weights(self.target_weights)
```

## Best Practices

1. **Avoid look-ahead bias**: use only data from the current callback arguments
2. **Account for costs**: Test with realistic commission and slippage
3. **Validate engine profile behavior**: check `vectorbt`, `backtrader`, and `zipline` profiles
4. **Size positions appropriately**: Don't risk more than 1-2% per trade
