# Order Types

ML4T Backtest supports multiple order types for realistic simulation.

Orders are submitted from a strategy using `broker.submit_order(...)`.

## Market Orders

```python
from ml4t.backtest.types import OrderType
```

Execute at market according to the configured execution mode/profile:

```python
broker.submit_order("AAPL", 100, order_type=OrderType.MARKET)   # buy
broker.submit_order("AAPL", -100, order_type=OrderType.MARKET)  # sell
```

## Limit Orders

Execute only if price reaches the limit:

```python
broker.submit_order(
    "AAPL",
    100,
    order_type=OrderType.LIMIT,
    limit_price=99.50,
)
```

- **Buy limit**: Fills if price drops to or below limit
- **Sell limit**: Fills if price rises to or above limit

## Stop Orders

Trigger a market order when stop price is reached:

```python
broker.submit_order(
    "AAPL",
    -100,
    order_type=OrderType.STOP,
    stop_price=95.00,
)
```

- **Buy stop**: Triggers when price rises to stop (breakout entry)
- **Sell stop**: Triggers when price falls to stop (stop loss)

## Trailing Stop Orders

Trailing stops dynamically update the stop level as price moves favorably.

```python
broker.submit_order(
    "AAPL",
    -100,
    order_type=OrderType.TRAILING_STOP,
    trail_amount=2.50,
)
```

## Order Processing

Orders are processed in **exit-first** order:

1. Stop losses and take profits (exits)
2. New position entries

This matches real broker behavior and prevents unrealistic fills.

## Order Management

```python
broker.get_order(order_id)
broker.get_pending_orders()
broker.update_order(order_id, limit_price=100.0)
broker.cancel_order(order_id)
broker.close_position("AAPL")

# Bracket order helper: entry + take profit + stop loss
broker.submit_bracket("AAPL", quantity=100, take_profit=110, stop_loss=95)
```
