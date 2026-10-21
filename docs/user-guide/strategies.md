# Strategies

Every strategy subclasses `Strategy` and implements `on_data`. The broker is your interface to the market -- use it to submit orders, query positions, and set risk rules.

The `MyStrategy` class sketches the callback interface. Later `broker` calls belong inside a callback; the [quickstart](../getting-started/quickstart.md) supplies a complete, checked run.

## Strategy Interface

```python
from ml4t.backtest import Strategy

class MyStrategy(Strategy):
    def on_prepare(self, broker, config=None):
        """Called once with resolved configuration and no future feed data. Optional."""
        pass

    def on_start(self, broker):
        """Called once before the first bar. Optional.

        Use this to set position rules, initialize state, or log config.
        """
        pass

    def on_data(self, timestamp, data, context, broker):
        """Called on every bar. Required.

        Args:
            timestamp: Current bar's datetime
            data: Dict of {asset: bar_dict}
            context: Dict of context data (if provided via DataFeed)
            broker: Broker for orders and position queries
        """
        pass

    def on_end(self, broker):
        """Called once after the last bar. Optional.

        Use this for final cleanup or logging.
        """
        pass
```

The callback order is `on_start`, `on_prepare`, one `on_data` call for each accepted
market event, and finally `on_end`. `on_prepare` receives the resolved config but no
future feed timestamps. Use `broker.register_target_intent(...)` during `on_prepare`
for a portfolio decision that was made before an opening auction. The
[Execution Semantics](execution-semantics.md#causal-strategy-lifecycle) reference
defines cutoff validation, opening fills, and position-rule activation.

### What `data` Contains

```python
data = {
    "AAPL": {
        "open": 150.0,
        "high": 152.0,
        "low": 149.5,
        "close": 151.0,
        "volume": 1_000_000,
        "signals": {"prediction": 0.85, "momentum": 0.12},
    },
    "MSFT": { ... },
}
```

Signals come from the optional signals DataFrame passed to DataFeed. Access them via `bar.get("signals", {}).get("column_name")`.

## Use causal history and session callbacks

These additional runtime services are development-source features, outside the
preserved 0.1.13 release candidate.

Pass a `RuntimeServices` instance to the existing `Engine`. The strategy's `context`
remains the input feed context. Rows before `start` supply history but cannot trade
or create account observations. The service becomes unavailable after the run.

This example requires actual timestamped closing bars and Specs 0.2.0. It does
not reinterpret midnight session labels as close instants. Runtime services
require a declared `event_time` or `bar_close` timestamp role. Convert session
labels to physical instants before using these services.

<!-- ml4t-doc-test: runtime-causal-session -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.specs import FeedSpec, LifecyclePhase, TimestampSemantics
from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy

dates = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [2, 3, 6]]
prices = pl.DataFrame({
    "timestamp": dates,
    "asset": ["TEST"] * 3,
    "close": [100.0, 101.0, 102.0],
})
services = RuntimeServices(start=datetime(2025, 1, 6, tzinfo=UTC))
trace = []

def before(timestamp, data, context, broker):
    history = services.market.history(["TEST"], ["close"], bar_count=2)
    assert history["close"].to_list() == [100.0, 101.0]
    assert data == {} and context == {}
    trace.append(("before", timestamp))

def at_close(timestamp, data, context, broker):
    assert services.market.current(["TEST"], ["close"])["close"].to_list() == [102.0]
    trace.append(("close", timestamp))

services.scheduler.register(
    before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
)
services.scheduler.register(at_close, phase=LifecyclePhase.CLOSE, at="session_close")

class Observe(Strategy):
    def on_data(self, timestamp, data, context, broker):
        trace.append(("data", timestamp))

spec = FeedSpec(time_contract_version=2, calendar="NYSE", timezone="UTC",
                data_frequency="1d", timestamp_semantics=TimestampSemantics.BAR_CLOSE,
                bar_label="right")
result = Engine(DataFeed(prices_df=prices, feed_spec=spec), Observe(),
                BacktestConfig(initial_cash=10000), runtime=services).run()
assert trace == [("before", datetime(2025, 1, 6, 13, 45, tzinfo=UTC)),
                 ("data", dates[-1]), ("close", dates[-1])]
assert result.to_equity_dataframe()["timestamp"].to_list() == [dates[-1]]
print("Runtime callbacks and causal warmup verified.")
```

History's `bar_count` counts observed price instants, retaining missing asset
rows. `known_at_cols={"field": "publication_timestamp_column"}` masks fields until
available. `forward_fill=True` is an explicit read view with source timestamps;
it never adds executable quotes or volume. Missing raw volume remains unknown
in the view. `last_traded` requires actual positive observed volume;
`last_observed` works with quotes alone. `can_trade` checks the current quote,
not buying power or all order controls.

Register callbacks before execution or in `on_start`/`on_prepare`. Registrations
made during a run are removed afterward; registrations made beforehand persist
when reusing the service for a new Engine. Callbacks use the existing lifecycle
transaction and roll back broker mutations if they fail.

Pre-open callbacks are calendar events with empty data/context. They use only
previously available history and cannot fill orders immediately. Other scheduled
callbacks require an observed price event at the requested instant; the last
input row is never promoted to an official session close. Offsets count elapsed
minutes from the named endpoint. Calendar events can occur between sparse price
observations without adding price bars or account marks. A disabled calendar
rejects session schedules.

Opening-auction callbacks require `event_time` opening observations. Completed
bars cannot supply an opening quote. Where continuous sessions meet at the same
instant, a right-labeled closing bar and its close callback run before the next
session's pre-open callback. That callback can read the bar that just closed.

## Broker Methods

### Submitting Orders

```python
# Market order: buy 100 shares
broker.submit_order("AAPL", 100)

# Short sell: negative quantity
broker.submit_order("AAPL", -100)

# Limit order
from ml4t.backtest.types import OrderType
broker.submit_order("AAPL", 100, order_type=OrderType.LIMIT, limit_price=150.0)

# Stop order
broker.submit_order("AAPL", 100, order_type=OrderType.STOP, stop_price=145.0)

# Stop-limit order
broker.submit_order("AAPL", 100, order_type=OrderType.STOP_LIMIT,
                    stop_price=145.0, limit_price=144.0)

# Market-on-close order
broker.submit_order("AAPL", 100, order_type=OrderType.MOC)
```

### Bracket Orders

Submit entry + take-profit + stop-loss in one call:

```python
orders = broker.submit_bracket(
    asset="AAPL",
    quantity=100,
    take_profit=165.0,  # TP target price
    stop_loss=145.0,    # SL price
)
# Returns (entry_order, tp_order, sl_order) or None if rejected
```

### Querying Positions

```python
# Single position
pos = broker.get_position("AAPL")
if pos is not None:
    print(pos.quantity, pos.entry_price, pos.bars_held)

# All positions
for asset, pos in broker.get_positions().items():
    print(f"{asset}: {pos.quantity} shares @ ${pos.entry_price:.2f}")

# Account value and cash
equity = broker.get_account_value()
cash = broker.get_cash()
```

### Closing Positions

```python
# Close a specific position
broker.close_position("AAPL")

# Close at the current bar close
broker.close_position("AAPL", order_type=OrderType.MOC)
```

### Rebalancing

```python
# Target weight allocation
broker.rebalance_to_weights({
    "AAPL": 0.30,   # 30% of portfolio
    "MSFT": 0.30,
    "GOOG": 0.20,
    # Remaining 20% stays in cash
})
```

### Order Management

```python
# Cancel a pending order
broker.cancel_order(order_id)

# Check rejected orders
rejected = broker.get_rejected_orders()
for order in rejected:
    print(f"{order.asset}: {order.rejection_reason}")
```

## Setting Risk Rules

Set position rules in `on_start`. They apply automatically on every bar:

```python
from ml4t.backtest import StopLoss, TakeProfit, TrailingStop, RuleChain

class ProtectedStrategy(Strategy):
    def on_start(self, broker):
        broker.set_position_rules(RuleChain([
            StopLoss(pct=0.05),
            TakeProfit(pct=0.15),
            TrailingStop(pct=0.03),
        ]))
```

See [Risk Management](risk-management.md) for the full rule catalog and composition patterns.

## Strategy Templates

The library includes four ready-to-use templates. Subclass them and override the decision methods.

### SignalFollowingStrategy

For ML predictions, technical indicators, or any pre-computed signal:

```python
from ml4t.backtest.strategies.templates import SignalFollowingStrategy

class MyMLStrategy(SignalFollowingStrategy):
    signal_column = "rf_prediction"  # Column name in signals DataFrame
    position_size = 0.05             # 5% of equity per position

    def should_enter_long(self, signal):
        return signal > 0.7

    def should_exit(self, signal):
        return signal < 0.3
```

### MomentumStrategy

Enter on positive momentum, exit on negative:

```python
from ml4t.backtest.strategies.templates import MomentumStrategy

class MyMomentum(MomentumStrategy):
    lookback = 60            # 60-bar momentum window
    entry_threshold = 0.10   # Enter on 10% return
    exit_threshold = 0.0     # Exit when momentum turns negative
    position_size = 0.10     # 10% of equity
```

### MeanReversionStrategy

Buy oversold, sell at reversion:

```python
from ml4t.backtest.strategies.templates import MeanReversionStrategy

class MyMeanReversion(MeanReversionStrategy):
    lookback = 30
    entry_zscore = -2.5   # Enter when 2.5 std below mean
    exit_zscore = 0.5     # Exit when 0.5 std above mean
    position_size = 0.10
```

### LongShortStrategy

Rank assets and go long top N, short bottom N:

```python
from ml4t.backtest.strategies.templates import LongShortStrategy

class MyLongShort(LongShortStrategy):
    signal_column = "alpha_score"
    long_count = 10
    short_count = 10
    position_size = 0.05
    rebalance_frequency = 21  # Monthly rebalance
```

## Patterns

### Multi-Asset with Position Sizing

```python
class SizedStrategy(Strategy):
    max_positions = 10

    def on_data(self, timestamp, data, context, broker):
        positions = broker.get_positions()
        if len(positions) >= self.max_positions:
            return

        equity = broker.get_account_value()
        per_position = equity / self.max_positions

        for asset, bar in data.items():
            if broker.get_position(asset) is not None:
                continue
            signal = bar.get("signals", {}).get("score", 0)
            if signal > 0.7:
                shares = int(per_position / bar["close"])
                if shares > 0:
                    broker.submit_order(asset, shares)
```

### Per-Asset Risk Rules

```python
class AssetSpecificRules(Strategy):
    def on_start(self, broker):
        # Different rules for different assets
        broker.set_position_rules(
            RuleChain([StopLoss(pct=0.03), TrailingStop(pct=0.02)]),
            asset="AAPL",
        )
        broker.set_position_rules(
            RuleChain([StopLoss(pct=0.10), TakeProfit(pct=0.20)]),
            asset="BTC",
        )
        # Global fallback for any other asset
        broker.set_position_rules(StopLoss(pct=0.05))
```

## Best Practices

1. **Avoid look-ahead bias** -- only use data from the current bar and earlier
2. **Account for costs** -- test with realistic commission and slippage
3. **Size positions conservatively** -- don't risk more than 1-2% per trade
4. **Use NEXT_BAR mode** -- for production strategies, avoid SAME_BAR
5. **Validate with profiles** -- compare results across framework profiles

## In the book

Chapter 16, Section 16.3, [Single asset ml4t-backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/04_single_asset_ml4t_backtest.ipynb) implements an event-driven strategy and reconciles its trade log. This page describes the reusable `Strategy` interface.

## Next Steps

- [Book Guide](../book-guide/index.md) -- chapter and case-study map for strategy workflows
- [Stateful Strategies](stateful-strategies.md) -- advanced patterns that require event-driven execution
- [Risk Management](risk-management.md) -- full rule catalog and composition
- [Order Types](orders.md) -- limit, stop, bracket orders in detail
- [Data Feed](data-feed.md) -- how data and signals are structured
