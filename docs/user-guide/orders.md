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

Execute at the current eligible bar's close.

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

### Expire orders at a session close

This policy is a development-source feature outside the preserved 0.1.13 release
candidate. The default `cancel_policy="never"` retains pending orders across
sessions. Choose `cancel_policy="session_end"` to expire only the unfilled
remainder at its assigned calendar session's close. A closing quote can fill the
order first. `MOC` retains its current-bar meaning; submit it at the actual
session endpoint when that is the desired execution time.

<!-- ml4t-doc-test: session-end-cancel -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.types import OrderStatus

dates = [datetime(2025, 1, 2, 20, 55, tzinfo=UTC),
         datetime(2025, 1, 3, 14, 30, tzinfo=UTC)]
prices = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 2,
                       "close": [100.0, 100.0]})
orders = []

class Submit(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp == dates[0]:
            orders.append(broker.submit_order("TEST", 1))

result = Engine(DataFeed(prices_df=prices,
                        feed_spec={"timestamp_semantics": "event_time"}),
                Submit(), BacktestConfig(calendar="NYSE",
                                          cancel_policy="session_end")).run()
assert result.to_fills_dataframe().is_empty()
assert orders[0].status is OrderStatus.CANCELLED
assert orders[0].cancelled_at == datetime(2025, 1, 2, 21, tzinfo=UTC)
assert orders[0].cancellation_reason == "session_end"
assert result.to_equity_dataframe()["timestamp"].to_list() == dates
print("The prior-session order expired before the next opening quote.")
```

The policy requires a named calendar and declared `event_time` or `bar_close`
instants. Session labels must first be converted to physical clocks. New orders
must belong to an active session or its scheduled pre-open. Pre-open orders are
not eligible before that callback's target session opens. A right-labeled bar at
the opening instant still belongs to the previous interval and cannot fill an
upcoming-session order.

With sparse inputs, expiration is processed as the simulation clock advances,
with `cancelled_at` retaining the official session close. No closing price is
invented. A declared `context_df` clock at the close can advance expiration when
no price bar exists. If neither price nor scheduled/context clocks reach the
close, the pending order remains open. An `end` filter alone does not create a
closing observation. User cancellations retain `cancellation_reason="user"` and
the current clock when available.

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

In development source, a triggered stop-limit stays a limit order until filled,
cancelled or expired. A gap beyond its limit does not execute it. A triggered
stop's partial remainder stays market eligible without needing to touch its stop
again. Changing `stop_price` rearms the trigger; changing only its limit or
remaining quantity preserves activation. `order.stop_triggered_at` records the
observed trigger instant, which is also retained on subsequent fills.

### Conditional price methods

The native `conditional_order_mode="bar_ohlc"` uses bar extrema for triggers,
fills limits at their limit level and fills newly triggered stops at the stop or
gap-open price. When a stop activates intrabar and its limit requires a reversal,
an earlier low/high cannot prove execution after activation. The order waits
unless the opening gap or closing observation establishes that sequence.

Select `"execution_price"` to use the configured execution observation for both
triggers and fills. For example, deferred close execution tests the close alone,
without using an intrabar high to trigger a stop. This method can improve a limit
fill when that observation is better than its limit. Both methods enforce the
limit against the final price after impact and slippage. An inadmissible price
leaves the order pending and consumes no fees or participation.

<!-- ml4t-doc-test: conditional-stop-limit -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy, OrderType
from ml4t.backtest.config import ConditionalOrderMode, ExecutionMode, ExecutionPrice

times = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [6, 7, 8, 10]]
prices = pl.DataFrame({"timestamp": times, "asset": ["TEST"] * 4,
    "open": [100.0, 110.0, 103.0, 99.0], "high": [101.0, 112.0, 104.0, 100.0],
    "low": [99.0, 109.0, 101.0, 98.0], "close": [100.0, 111.0, 102.0, 99.0]})

class Buy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp == times[0]:
            broker.submit_order("TEST", 10, order_type=OrderType.STOP_LIMIT,
                                stop_price=105.0, limit_price=104.0)

result = Engine(DataFeed(prices_df=prices), Buy(), BacktestConfig(
    calendar=None, initial_cash=10000, execution_mode=ExecutionMode.NEXT_BAR,
    next_bar_execution_price=ExecutionPrice.CLOSE,
    conditional_order_mode=ConditionalOrderMode.EXECUTION_PRICE)).run()
fills = result.to_fills_dataframe()
assert fills["price"].to_list() == [102.0]
assert fills["timestamp"].to_list() == [times[2]]
assert fills["stop_triggered_at"].to_list() == [times[1]]
assert result.to_portfolio_state_dataframe()["cash"].to_list() == [10000, 10000, 8980, 8980]
assert result.to_equity_dataframe()["equity"].to_list() == [10000, 10000, 10000, 9970]
print("Stop-limit activation, execution and account values verified.")
```

This four-session close-price case was also run through published Zipline
Reloaded 3.2.0 with `StopLimitOrder(104, 105)`, zero commission and zero impact.
Its fill, cash and NAV agree. Other callback, account and slippage settings need
their own qualification.

Conditional orders require finite positive price parameters. Invalid prices and
nonfinite, Boolean or nonnumeric quantities fail before consuming an order ID or
changing the account. Zero quantity remains a no-op.

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
