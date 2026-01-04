# Order Types

ML4T Backtest supports multiple order types for realistic simulation.

## Market Orders

Execute at the next bar's open price:

```python
self.buy(size=100)  # Market buy
self.sell(size=100)  # Market sell
```

## Limit Orders

Execute only if price reaches the limit:

```python
from ml4t.backtest import OrderType

self.buy(size=100, price=99.50, order_type=OrderType.LIMIT)
```

- **Buy limit**: Fills if price drops to or below limit
- **Sell limit**: Fills if price rises to or above limit

## Stop Orders

Trigger a market order when stop price is reached:

```python
self.buy(size=100, stop=101.00, order_type=OrderType.STOP)
```

- **Buy stop**: Triggers when price rises to stop (breakout entry)
- **Sell stop**: Triggers when price falls to stop (stop loss)

## Stop-Limit Orders

Trigger a limit order when stop is reached:

```python
self.buy(
    size=100,
    stop=101.00,
    price=101.50,
    order_type=OrderType.STOP_LIMIT
)
```

## Order Processing

Orders are processed in **exit-first** order:

1. Stop losses and take profits (exits)
2. New position entries

This matches real broker behavior and prevents unrealistic fills.

## Order Management

```python
# Set stop loss after entry
self.set_stop(price=95.00)

# Set profit target
self.set_target(price=110.00)

# Cancel all open orders
self.cancel_all()
```
