# Order Types

Orders are submitted from a strategy via `broker.submit_order()`. The default order type is `MARKET`.

Run the [order and timing tutorial](../tutorials/orders-and-timing.md) to compare fills on one fixed price panel.

The order calls below assume `broker` is the argument passed to a strategy callback. Run the linked orders tutorial for a complete strategy, feed, and fill records.

## Market Orders

Execute at the configured execution price (open or close, depending on profile):

```python
# Buy 100 shares (positive quantity = buy)
broker.submit_order("AAPL", 100)

# Sell/short 100 shares (negative quantity = sell)
broker.submit_order("AAPL", -100)
```

In `NEXT_BAR` mode (default), ordinary market orders fill at the next eligible asset bar's open. In `SAME_BAR` mode, they use the current bar's configured `execution_price`; choose `ExecutionPrice.CLOSE` for a current-close comparison.

## Market-On-Close Orders

Execute at the current session close.

```python
from ml4t.backtest.types import OrderType

# Enter at the current bar close
broker.submit_order("AAPL", 100, order_type=OrderType.MOC)

# Flatten an open position at the current bar close
broker.close_position("AAPL", order_type=OrderType.MOC)
```

`MOC` is the one order type that overrides standard `NEXT_BAR` timing. In `NEXT_BAR`
mode, orders submitted during `on_data()` normally fill at the next bar's open, but
`MOC` orders fill at the current bar's close after strategy logic finishes. In
`SAME_BAR` mode, `MOC` also fills at the current bar's close.

With an exact-timestamp feed, this models a market-on-close fill at the current daily bar. With `DataFeed(session_col=...)`, a session decision occurs after every component bar has closed; a new `MOC` order waits for that asset's next bar close.

## Limit Orders

Execute only if price reaches the limit level:

```python
from ml4t.backtest.types import OrderType

# Buy limit: fills if price drops to $149.50 or below
broker.submit_order("AAPL", 100, order_type=OrderType.LIMIT, limit_price=149.50)

# Sell limit: fills if price rises to $155.00 or above
broker.submit_order("AAPL", -100, order_type=OrderType.LIMIT, limit_price=155.00)
```

Limit orders remain pending until filled or cancelled.

## Stop Orders

Trigger a market order when the stop price is reached:

```python
# Sell stop: triggers when price falls to $145.00 (protective stop)
broker.submit_order("AAPL", -100, order_type=OrderType.STOP, stop_price=145.00)

# Buy stop: triggers when price rises to $160.00 (breakout entry)
broker.submit_order("AAPL", 100, order_type=OrderType.STOP, stop_price=160.00)
```

## Stop-Limit Orders

Trigger a limit order when the stop price is reached:

```python
# When price falls to $145, place a limit sell at $144
broker.submit_order(
    "AAPL", -100,
    order_type=OrderType.STOP_LIMIT,
    stop_price=145.00,
    limit_price=144.00,
)
```

## Trailing Stop Orders

Dynamic stop that follows price movement:

```python
# Trail $2.50 below current price
broker.submit_order(
    "AAPL", -100,
    order_type=OrderType.TRAILING_STOP,
    trail_amount=2.50,
)
```

## Bracket Orders

Submit an entry with automatic take-profit and stop-loss exits:

```python
result = broker.submit_bracket(
    asset="AAPL",
    quantity=100,
    take_profit=165.00,
    stop_loss=145.00,
)

if result is not None:
    entry_order, tp_order, sl_order = result
```

The exit side is automatically determined from the entry direction. Price validation warns if take-profit is below entry (for longs) or stop-loss is above entry.

## Order Lifecycle

1. **Submitted** -- order is queued in the OrderBook
2. **Validated** -- Gatekeeper checks cash/margin constraints
3. **Filled** -- FillExecutor applies slippage and commission
4. **Rejected** -- insufficient cash, short selling not allowed, etc.

Check rejected orders:

```python
rejected = broker.get_rejected_orders()
for order in rejected:
    print(f"{order.asset}: {order.rejection_reason}")
```

## Order Processing Sequence

On each bar, pending orders are processed based on the `fill_ordering` config:

| Mode | Sequence |
|------|----------|
| `EXIT_FIRST` | All exits, then all entries (default) |
| `FIFO` | Submission order |
| `SEQUENTIAL` | Submission order, no exit/entry separation |

See [Execution Semantics](execution-semantics.md#fill-ordering) for details.

## Closing Positions

```python
# Close a single position
broker.close_position("AAPL")

# Close at the current session close
broker.close_position("AAPL", order_type=OrderType.MOC)

# Cancel a pending order by ID
broker.cancel_order(order.id)
```

## In the book

Chapter 16, Section 16.3, [Single asset ml4t-backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/04_single_asset_ml4t_backtest.ipynb) reconciles submitted decisions with completed fills. The [engine divergence notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/07_engine_divergence_anatomy.ipynb) in the companion varies execution assumptions.

## Next Steps

- [Execution Semantics](execution-semantics.md) -- how orders fill and at what price
- [Risk Management](risk-management.md) -- automatic stop-loss and trailing stop rules
- [Strategies](strategies.md) -- order patterns in strategy code
