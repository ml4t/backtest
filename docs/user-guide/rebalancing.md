# Rebalancing

For multi-asset strategies that target portfolio weights, the broker provides `rebalance_to_weights()` and the execution module provides a `TargetWeightExecutor` for advanced control.

## Simple Rebalancing

```python
class EqualWeightStrategy(Strategy):
    def __init__(self, assets, rebalance_interval=21):
        self.assets = assets
        self.rebalance_interval = rebalance_interval
        self.bar_count = 0

    def on_data(self, timestamp, data, context, broker):
        self.bar_count += 1
        if self.bar_count % self.rebalance_interval != 1:
            return

        n = len(self.assets)
        weights = {asset: 1.0 / n for asset in self.assets}
        broker.rebalance_to_weights(weights)
```

`rebalance_to_weights()` computes the delta between current holdings and target weights, then submits sell orders (to reduce overweight positions) before buy orders (to fill underweight positions).

## Rebalance Modes

The `rebalance_mode` config controls how portfolio value is computed during rebalancing:

| Mode | Behavior | Matches |
|------|----------|---------|
| `SNAPSHOT` | Freeze portfolio value at start of rebalance. All targets use the same base. | Backtrader `order_target_percent` |
| `INCREMENTAL` | Recompute portfolio value after each fill. Most accurate cash tracking. | Default |
| `HYBRID` | Freeze value for target computation, but fill sequentially with live cash checks. | VectorBT default |

```python
from ml4t.backtest.config import RebalanceMode

config = BacktestConfig(rebalance_mode=RebalanceMode.SNAPSHOT)
```

## Rebalance Headroom

The `rebalance_headroom_pct` parameter scales target weights to leave a cash buffer:

```python
config = BacktestConfig(rebalance_headroom_pct=0.998)
# Targets 99.8% of computed weights, leaving 0.2% cash buffer
```

This prevents rounding-induced over-allocation. Backtrader uses 0.998 by default.

## Late Assets and Missing Prices

When assets start trading at different times (e.g., IPOs), two parameters control behavior:

```python
from ml4t.backtest.config import LateAssetPolicy, MissingPricePolicy

config = BacktestConfig(
    # Require 2 bars of history before trading an asset
    late_asset_policy=LateAssetPolicy.REQUIRE_HISTORY,
    late_asset_min_bars=2,

    # Use last known price when current bar is missing
    missing_price_policy=MissingPricePolicy.USE_LAST,
)
```

## Advanced: TargetWeightExecutor

For more control, use the `TargetWeightExecutor` with a `RebalanceConfig`:

```python
from ml4t.backtest.execution.rebalancer import TargetWeightExecutor, RebalanceConfig

rebalance_config = RebalanceConfig(
    min_trade_value=100,        # Optional: skip trades smaller than $100
    min_weight_change=0.01,     # Optional: skip changes smaller than 1%
    allow_fractional=False,     # Round to whole shares
    max_single_weight=0.25,     # Cap any single position at 25%
    cancel_before_rebalance=True,
)

executor = TargetWeightExecutor(rebalance_config)
```

The executor integrates with external portfolio optimizers (riskfolio-lib, PyPortfolioOpt, cvxpy) through the `WeightProvider` protocol:

```python
class MyOptimizer:
    def get_weights(self, data, broker):
        # Your optimization logic here
        return {"AAPL": 0.3, "MSFT": 0.3, "GOOG": 0.4}
```

By default, `RebalanceConfig` uses `min_trade_value=0.0` and
`min_weight_change=0.0`, so no trade-size or weight-delta filter is applied
unless you opt into one explicitly.

### Schedule metadata

Session-based schedules need enough metadata to identify exchange session closes. Weekly and
month-end schedules require `calendar`. Daily feeds must also set `data_frequency="daily"`.
Intraday feeds must set `calendar`, `timezone`, `data_frequency`, and `timestamp_semantics`. Set
`session_start_time` for a custom evening boundary. Morning-start sessions use the local calendar
date, so a morning value may only restate the configured calendar's standard open. Intraday feeds
without calendar boundary metadata may instead pass `is_session_close=True` or `False` on every
event. Under this option, session identity uses the timestamp's raw date, so it is valid only when
sessions do not cross local midnight. Overnight sessions must provide `data_frequency` and
`timestamp_semantics`. Do not mix explicit and inferred boundaries on one date. Without either
form of intraday metadata, events less than 12 hours apart are rejected.

```python
from ml4t.backtest import RebalanceSchedule
from ml4t.backtest.execution import RebalanceConfig

rebalance_config = RebalanceConfig(
    schedule=RebalanceSchedule.fixed_n_sessions(5),
    calendar="CME_Equity",
    timezone="UTC",
    session_start_time="17:00",
    data_frequency="1m",
    timestamp_semantics="bar_close",
)
```

Naive timestamps use `timezone` as their source timezone. Session boundaries are then applied in
the exchange timezone. Weekly and month-end schedules fire only on the final scheduled exchange
session in the period. If the feed advances into a later period without the completed period's
final exchange session, evaluation raises a feed-completeness error. An incomplete final period
does not move the rebalance to an earlier bar.

For intraday schedules, every complete exchange session that requires a rebalance must contain a
timestamp aligned with the exchange close. Batch and online evaluation identify the first complete
required session whose close was reached but never matched. Exchange holidays and a final session
whose data ends before its close are not alignment failures.

Explicit timestamp schedules match absolute instants. A naive scheduled timestamp uses the
configured `timezone`, so it can match an aware feed timestamp expressed in another timezone. If a
scheduled instant falls between the first and last observed feed instants but no event matches it,
batch resolution and final online validation raise an alignment error. Scheduled instants before
or after the observed window are ignored, including a schedule earlier than the first event on a
partially observed day. Online evaluation requires chronological events and rejects a timestamp
that moves backward.

`TargetWeightExecutor.prepare_schedule()` was removed before 0.1.0 because it required the future
feed timestamp sequence. Put the metadata shown above on `RebalanceConfig`; `should_rebalance()`
then evaluates only the current event and prior evaluator state. Call `should_rebalance()` or
scheduled `execute()` for every event in chronological order, including warm-up periods. After the
final event, call `validate_completed_run()` so a missing close in the final session is reported.
When scheduled `execute()` receives an Engine broker, the Engine registers and runs this final
validation automatically. Standalone use and direct `should_rebalance()` calls still require the
explicit completion call. Gating `execute()` on `should_rebalance()` for the same timestamp is
supported when both calls receive the same `is_session_close`, or when only `should_rebalance()`
receives it. `every_bar` schedules impose no event-coverage requirement. Other conditional skips
fail completion validation because session counters would be incomplete.
Call `reset()` before reusing an executor for another run or changing its public
`RebalanceConfig`. Backward session dates and mid-run configuration changes raise instead of
silently restarting session counters.
The engine validates `LongShortStrategy` schedule alignment after `on_end`, independently of any
callback override.

## See It in Action

The [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) book uses TargetWeightExecutor extensively:

- **Ch16 case studies** — all 6 Engine-based cases (ETFs, FX, equities, crypto, futures, options) use TargetWeightExecutor for ML prediction → portfolio weight → rebalance
- **Ch17** (`portfolio_construction`) — portfolio optimization with weight constraints

The common pattern: ML model generates predictions, predictions are converted to portfolio weights, TargetWeightExecutor handles the order generation and execution.

## Next Steps

- [Book Guide](../book-guide/index.md) -- portfolio-construction and case-study mapping
- [Strategies](strategies.md) -- strategy patterns and templates
- [Configuration](configuration.md) -- rebalance-related config parameters
- [Results & Analysis](results.md) -- analyze portfolio outcomes
