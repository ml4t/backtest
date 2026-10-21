# Execution Semantics

This page describes when orders are eligible to fill and how the engine processes them. Configuration controls the listed execution choices; the lifecycle and causal ordering rules remain part of the engine contract.

Run the [order and timing tutorial](../tutorials/orders-and-timing.md) to compare fills on one fixed price panel.

The configuration fragments below change one execution choice at a time and assume their enum imports. Run the linked orders tutorial to inspect the resulting fills.

## Execution Timing

### NEXT_BAR (Default)

Orders submitted during `on_data()` are queued and filled at the **next bar's open** price.

```
Bar N:  Strategy sees close=$100, submits buy order
Bar N+1: Order fills at open=$101
```

This is the realistic model -- your strategy decides based on today's information and the trade executes at tomorrow's opening auction.

```python
from ml4t.backtest.config import BacktestConfig
from ml4t.backtest.types import ExecutionMode

config = BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR)  # default
```

`OrderType.MOC` is the exception. In `NEXT_BAR` mode, `MOC` orders submitted during
`on_data()` still fill on the current bar, at the close, after strategy logic runs. For session-level decisions with `DataFeed(session_col=...)`, an `MOC` order instead waits for the next eligible bar close for that asset.

### Causal Strategy Lifecycle

The engine invokes `on_start(broker)`, `on_prepare(broker, config)`, `on_data(...)` once for each
accepted market event, and `on_end(broker)`. All callbacks pass through the versioned lifecycle
dispatcher. `on_prepare` receives calendar and execution configuration, but it does not receive the
feed's future timestamps. Built-in rebalance schedules evaluate the current event against calendar
metadata instead of resolving the complete run schedule before the run starts.

The former `on_before_risk()` callback was removed because its backtest-only position in the bar
cycle could not be reproduced by a live engine. A strategy that still defines it fails during
engine construction, before the broker or account is created. Move decisions based on an accepted
market event to `on_data()`. Under `NEXT_BAR`, an ordinary order submitted by `on_data()` becomes
eligible at the next bar's open. It cannot fill at the current bar's open after the strategy has
observed that bar's completed values.

Initial and scheduled opening-auction target intents use the separate pre-open contract. They must
be decided from information available before the auction cutoff and do not use `on_data()` to
recover an earlier fill price.

Register an initial target during causal initialization. The target records the decision and
information cutoff, but quantity lowering waits for the eligible opening price:

```python
from datetime import UTC, date, datetime

from ml4t.specs import (
    AssetTarget,
    CanonicalTargetIntent,
    IntentReason,
    LifecyclePhase,
    ResidualPolicy,
    RoundingPolicy,
    TargetMeasure,
)


def on_prepare(self, broker, config=None):
    decision = datetime(2026, 8, 2, 20, 0, tzinfo=UTC)
    broker.register_target_intent(
        CanonicalTargetIntent(
            intent_id="initial-portfolio",
            decision_time=decision,
            information_cutoff=decision,
            effective_session=date(2026, 8, 3),
            effective_phase=LifecyclePhase.PRE_OPEN,
            targets=(AssetTarget("SPY", TargetMeasure.WEIGHT, 0.95),),
            idempotency_key="initial-portfolio-2026-08-03",
            measure=TargetMeasure.WEIGHT,
            cash_buffer=0.05,
            rounding=RoundingPolicy.TOWARD_ZERO,
            residual=ResidualPolicy.KEEP_CASH,
            reason=IntentReason.REBALANCE,
        )
    )
```

The engine validates the cutoff, lowers the target at the opening auction, records canonical child
intent and order lineage, reconciles fills and remaining quantity, then applies each target's
position rule to the resulting position. Opening child orders use OPG time-in-force: any quantity
not filled by the eligible opening auction is cancelled and is never carried into a later bar. A
scheduled target uses the same method from an earlier event with a future
`effective_session`. Same-session registration from `on_data()` is rejected because that callback
has already observed information after the opening phase.

Targets support two position-rule declaration modes. If no `AssetTarget` declares a policy, the
intent-level `position_rule_policy_id` applies to every asset and the existing single-rule
registration remains valid. If any target declares a policy, each target's declaration is
authoritative and the intent-level field must be absent. Pass a mapping containing exactly the
policy IDs referenced by that intent. A target without an ID has no target-managed rule.

<!-- ml4t-doc-test: preopen-mixed-rules -->
```python
from datetime import UTC, date, datetime

import polars as pl
from ml4t.specs import (
    AssetTarget,
    CanonicalTargetIntent,
    IntentReason,
    LifecyclePhase,
    ResidualPolicy,
    RoundingPolicy,
    TargetMeasure,
)

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.risk.position import StopLoss, TrailingStop

decision = datetime(2026, 8, 2, 20, 0, tzinfo=UTC)
mixed_target = CanonicalTargetIntent(
    intent_id="mixed-risk-portfolio",
    decision_time=decision,
    information_cutoff=decision,
    effective_session=date(2026, 8, 3),
    effective_phase=LifecyclePhase.PRE_OPEN,
    targets=(
        AssetTarget(
            "SPY",
            TargetMeasure.WEIGHT,
            0.40,
            position_rule_policy_id="spy-stop-5",
        ),
        AssetTarget(
            "QQQ",
            TargetMeasure.WEIGHT,
            0.40,
            position_rule_policy_id="qqq-trail-8",
        ),
        AssetTarget("IWM", TargetMeasure.WEIGHT, 0.15),
    ),
    idempotency_key="mixed-risk-portfolio-2026-08-03",
    measure=TargetMeasure.WEIGHT,
    cash_buffer=0.05,
    rounding=RoundingPolicy.TOWARD_ZERO,
    residual=ResidualPolicy.KEEP_CASH,
    reason=IntentReason.REBALANCE,
)

class MixedRuleStrategy(Strategy):
    def on_prepare(self, broker, config=None):
        broker.register_target_intent(
            mixed_target,
            position_rules={
                "spy-stop-5": StopLoss(0.05),
                "qqq-trail-8": TrailingStop(0.08),
            },
        )

    def on_data(self, timestamp, data, context, broker):
        pass


assets = ("SPY", "QQQ", "IWM")
prices = pl.DataFrame(
    {
        "timestamp": [datetime(2026, 8, 3)] * len(assets),
        "asset": assets,
        "open": [100.0] * len(assets),
        "high": [101.0] * len(assets),
        "low": [99.0] * len(assets),
        "close": [100.0] * len(assets),
        "volume": [1_000_000.0] * len(assets),
    }
)
engine = Engine(
    DataFeed(prices_df=prices),
    MixedRuleStrategy(),
    BacktestConfig(retain_intent_history=True),
)
result = engine.run()

assert result.metrics["target_intent_count"] == 1
assert result.metrics["target_rule_reconciliation_count"] == 3
```

Registration validates the complete mapping before changing manager state. Missing
implementations, conflicting definitions, and unreferenced mapping keys reject the request.

`ExecutionPolicy.liquidity_fraction` applies only to opening-auction child orders. A value of 1.0
means no participation constraint, even when the requested quantity exceeds the bar volume. A
smaller value creates an opening-only participation limit unless the Engine receives an explicit
`execution_limits` object. Explicit limits must match the declared opening policy. Ordinary orders
use only the Engine's explicit `execution_limits` setting.

For a rule activated after an opening fill, `ExecutionPolicy.bar_path` controls daily OHLC
ambiguity. Use `REJECT_AMBIGUOUS` to fail when high-low order changes the result,
`OPEN_HIGH_LOW_CLOSE` or `OPEN_LOW_HIGH_CLOSE` to declare an order, or `CONSERVATIVE` to select the
more adverse supported outcome. Every result retains the lifecycle version, execution policy, and
intent record counts. Set `BacktestConfig(retain_intent_history=True)` to include full target,
child-intent, and reconciliation records in the result artifact. The default avoids copying an
unbounded event history into results from long-running or high-asset simulations. The broker
accessors and `export_target_intent_state()` remain available during and after a run.

Lifecycle callback counts are always retained for contract validation. Set
`BacktestConfig(retain_lifecycle_history=True)` only when a per-callback trace is needed for parity
analysis or debugging. The default does not retain one record per market event.

`ResidualPolicy.LARGEST_REMAINDER` is unsupported for fractional-share accounts regardless of the
rounding policy. Its allocation step adds whole units, so it cannot redistribute a rounded
fractional remainder, while an unrounded fractional target has no discrete remainder to allocate.
Use `KEEP_CASH`; unsupported combinations are rejected before opening orders are created or
restored.

A position rule associated with a target remains active until the position becomes flat or a later
target changes it. Repeating the same policy ID preserves the installed rule object, its state, and
its activation time. A different ID installs a fresh rule at that opening. A target with no
resolved ID removes the earlier target-managed rule and its activation context. These transitions
use the actual post-auction position, including a carried position after a partial fill, rejected
order, or zero quantity change. A failed new entry does not activate a rule. An explicit rule
override installed through the broker remains in place during target cleanup because cleanup
removes only the exact rule object installed by the target manager. Use
`get_target_rule_reconciliations()` to inspect the requested, prior, and resulting policy ID for
every asset target.

`export_target_intent_state()` does not contain account or order-book state. Restoring state that
already contains child orders therefore requires those orders and the corresponding account state
to exist in the destination broker. A fresh `Engine` cannot resume a post-opening state from the
target-intent payload alone. Unprocessed targets can be restored without order state.

### SAME_BAR

Orders can fill in the bar where they are submitted, using the configured `execution_price`. The default source is `OPEN`; set `ExecutionPrice.CLOSE` to model a current-close fill.

```
Bar N:  Strategy sees close=$100, submits buy order, fills at close=$100
```

Use this mode only when the comparison explicitly permits a decision and fill at the same close. It carries lookahead risk if the strategy observes that close before choosing an order that fills there. VectorBT supports several simulation modes, including [state-dependent order functions](https://vectorbt.dev/api/portfolio/base/); its timing cannot be summarized by this one setting.

```python
from ml4t.backtest import BacktestConfig, ExecutionMode
from ml4t.backtest.config import ExecutionPrice

config = BacktestConfig(
    execution_mode=ExecutionMode.SAME_BAR,
    execution_price=ExecutionPrice.CLOSE,
)
```

### Execution Price

The `execution_price` parameter controls which price source is used for market order fills:

| Value | Fill Price | Typical Use |
|-------|-----------|-------------|
| `OPEN` | Eligible bar open | Default next-bar fill |
| `CLOSE` | Eligible bar close | Same-bar close comparison |
| `VWAP` | Volume-weighted average | Requires volume data |
| `MID` | (high + low) / 2 | Simple approximation |
| `PRICE` | `FeedSpec.price_col` / `bar["price"]` | Custom reference price, derived bars |
| `BID` | Best bid quote | Passive or conservative sell-side marking |
| `ASK` | Best ask quote | Aggressive buy-side fills |
| `QUOTE_MID` | Quote midpoint | Microstructure-aware marking |
| `QUOTE_SIDE` | Ask for buys, bid for sells | Side-aware market execution |

`PRICE` is the default mark source and follows your feed schema. If you map `price_col="mid_price"`, then both `bar["price"]` and `ExecutionPrice.PRICE` use that midpoint.

`OrderType.MOC` does not use `execution_price`; it fills at the eligible bar's close. In session decision mode, that eligible bar is later than the decision.

### Mark Price

Open positions are marked independently of how market orders fill. `mark_price` uses the same `ExecutionPrice` enum as `execution_price`.

This is useful when you want to:

- fill orders at `QUOTE_SIDE` but mark inventory at `QUOTE_MID`
- trade from a synthetic `price_col` while keeping fills at `OPEN` or `CLOSE`
- mark long inventory conservatively on the bid and short inventory on the ask via `QUOTE_SIDE`

```python
from ml4t.backtest.config import ExecutionPrice

config = BacktestConfig(
    execution_price=ExecutionPrice.QUOTE_SIDE,
    mark_price=ExecutionPrice.QUOTE_MID,
)
```

If the requested quote field is unavailable, the broker falls back to the feed reference price and then to OHLC where applicable.

### Quote-Aware Backtests

When you provide bid/ask data and enable quote-aware execution or marking, the
backtest is quote-aware, not just OHLCV-aware.

That affects both execution and reporting:

- fills preserve the quote source and nullable quote context used for execution
- trades preserve entry and exit quote summaries
- portfolio state reflects the configured `mark_price`

This makes quote-side behavior auditable after the run instead of burying it in
aggregate PnL only.

## Fill Ordering

When multiple orders are pending on the same bar, the engine must decide the processing sequence. This matters because fills affect cash, which affects whether subsequent orders can be accepted.

### EXIT_FIRST (Default)

All exits process first (freeing cash), then all entries:

```
1. Process all exit orders → cash freed
2. Mark-to-market remaining positions
3. Process all entry orders → gatekeeper checks cash
```

This is the most capital-efficient ordering. Matches VectorBT's `call_seq='auto'`.

### FIFO

Orders process in submission order. Each order's gatekeeper check sees cash from all prior fills:

```
1. First submitted order fills → cash updated
2. Second submitted order fills → cash updated
3. ...
```

Matches Backtrader's submission-order processing.

### SEQUENTIAL

Orders process in submission order (typically alphabetical by asset) without exit/entry separation. Unlike EXIT_FIRST, exits do not pre-free cash for later entries.

Matches LEAN's per-order sequential buying-power model.

```python
from ml4t.backtest.config import FillOrdering

config = BacktestConfig(fill_ordering=FillOrdering.EXIT_FIRST)  # default
config = BacktestConfig(fill_ordering=FillOrdering.FIFO)
config = BacktestConfig(fill_ordering=FillOrdering.SEQUENTIAL)
```

### Entry Order Priority

When using EXIT_FIRST, entries are processed after exits. The `entry_order_priority` controls the sequence of entry orders:

| Value | Behavior |
|-------|----------|
| `SUBMISSION` | Keep strategy submission order (default) |
| `NOTIONAL_DESC` | Larger dollar entries first |
| `NOTIONAL_ASC` | Smaller dollar entries first |

## Stop and Take-Profit Execution

Position rules (StopLoss, TakeProfit, TrailingStop) are evaluated on every bar using OHLC data. Quote-aware execution changes market fills and position marking, but stop triggers still evaluate against bar data. The key question is: **at what price does a triggered stop fill?**

### Stop Fill Modes

The `stop_fill_mode` parameter controls stop/take-profit fill prices:

| Mode | Fill Price | Use Case |
|------|-----------|----------|
| `STOP_PRICE` | Exact stop/target level | Default, standard model |
| `CLOSE_PRICE` | Bar's close price | VectorBT with close-only data |
| `BAR_EXTREME` | Bar's low (stop) or high (TP) | Conservative/optimistic model |
| `NEXT_BAR_OPEN` | Next bar's open price | Zipline-style deferred exits |

**Gap handling.** If the bar opens beyond the stop level (a gap through the stop), the fill price is the bar's open, not the stop price. This accurately models gap risk:

```
Stop set at $95.00
Bar opens at $93.00 (gap down)
Fill price = $93.00 (open), not $95.00 (stop)
```

### Stop Level Basis

The `stop_level_basis` controls what price the stop percentage is calculated from:

| Value | Base Price | Use Case |
|-------|-----------|----------|
| `FILL_PRICE` | Actual fill price including slippage | Default, most frameworks |
| `SIGNAL_PRICE` | Close price when signal was generated | Backtrader behavior |

Example: You set `StopLoss(pct=0.05)`. With `FILL_PRICE`, the stop is 5% below where you actually got filled. With `SIGNAL_PRICE`, it's 5% below the close of the bar where you submitted the order.

## Trailing Stop Mechanics

Trailing stops track a "water mark" -- the highest price since entry (for longs) or lowest price since entry (for shorts) -- and exit when price retraces by a percentage from that mark.

### Water Mark Source

`trail_hwm_source` controls which price updates the water mark:

| Value | Update Price | Framework |
|-------|-------------|-----------|
| `CLOSE` | Bar's close | Default, most frameworks |
| `BAR_EXTREME` | Bar's high (longs) / low (shorts) | VectorBT Pro with OHLC |

### Initial Water Mark

`initial_hwm_source` controls the water mark on the entry bar:

| Value | Initial HWM | Framework |
|-------|------------|-----------|
| `FILL_PRICE` | Actual fill price | Default, event-driven frameworks |
| `BAR_CLOSE` | Entry bar's close | |
| `BAR_HIGH` | Entry bar's high | VectorBT Pro with OHLC |

### Trailing Stop Timing

`trail_stop_timing` controls when water marks update relative to the stop check. This is the subtlest parameter and the one that causes the most divergence between frameworks.

**LAGGED** (default): Check stop using the *previous* bar's water mark, then update water mark at end of current bar. This creates a 1-bar lag -- the stop can't trigger based on a new high set in the current bar.

**INTRABAR**: Update water mark before checking. The stop can trigger based on the current bar's extreme. More aggressive than LAGGED.

**VBT_PRO**: Two-pass algorithm matching VectorBT Pro exactly:

1. **Pass 1**: Check stop using the *previous* bar's water mark against the current bar's HIGH (long) or LOW (short)
2. If pass 1 doesn't trigger, update the water mark from the current bar's extreme
3. **Pass 2**: Check stop using the *updated* water mark against **CLOSE only** (not HIGH/LOW)

This precisely reproduces VBT Pro's `can_use_ohlc=False` behavior in the second pass.

```python
from ml4t.backtest.config import TrailStopTiming, WaterMarkSource, InitialHwmSource

# VectorBT Pro compatible trailing stops
config = BacktestConfig(
    trail_stop_timing=TrailStopTiming.VBT_PRO,
    trail_hwm_source=WaterMarkSource.BAR_EXTREME,
    initial_hwm_source=InitialHwmSource.BAR_HIGH,
)
```

## Commission and Slippage

## Quote Context on Fills

Every fill records the price source that was used along with nullable quote context:

- `price_source`
- `reference_price`
- `quote_mid_price`
- `bid_price`
- `ask_price`
- `spread`
- `bid_size`
- `ask_size`
- `available_size`

That data is available both in memory and in `result.to_fills_dataframe()` / `fills.parquet`, which makes it possible to audit quote-side behavior after the run.

Trade summaries preserve the same context at entry and exit, and
`result.to_portfolio_state_dataframe()` reflects the configured mark source for
each end-of-bar snapshot.

### Commission Models

| Type | Calculation | Config |
|------|------------|--------|
| `NONE` | No commission | `commission_rate=0` |
| `PERCENTAGE` | % of trade value | `commission_rate=0.001` (0.1%) |
| `PER_SHARE` | Fixed $ per share | `commission_per_share=0.005` |
| `PER_TRADE` | Fixed fee per fill | `commission_per_trade=5.0` |
| `PER_ORDER_PER_SHARE` | Cumulative per-order minimum on unit fees | `commission_per_share=0.005, commission_minimum=1.0` |
| `PER_ORDER` | Composed cumulative unit/notional/one-time fees | Explicit unit, rate, flat and minimum parameters |
| `TIERED` | Volume-based tiers | Custom model |

```python
from ml4t.backtest.config import CommissionType

# Interactive Brokers style
config = BacktestConfig(
    commission_type=CommissionType.PER_SHARE,
    commission_per_share=0.005,
    commission_minimum=1.0,
)
```

### Cumulative order fees and shared volume

These additional methods are development-source features. They are not part of the
preserved 0.1.13 release candidate.

`PER_SHARE` and `PER_TRADE` retain their native per-fill behavior. Select
`PER_ORDER_PER_SHARE` when one order's partial fills should share a minimum fee.
`PER_ORDER` additionally combines `commission_per_share`, `commission_rate`,
`commission_per_trade` and `commission_minimum`. Its notional component uses actual
fill prices and the declared contract multiplier; the flat component applies once.
Quotes use previously committed order quantities/notionals/fees and do not consume
a fill. Fees are committed only for actual executions.

`SharedVolumeParticipationLimit` shares one traded-volume budget across all orders
for the same asset and bar, in both directions. Missing/nonfinite traded volume
has no capacity; quote-side depth does not substitute for traded volume. Explicit
quote depth can further restrict a fill. `VolumeParticipationLimit` retains its
existing per-order convention. Neither option models a complete exchange order book.

The following run matches the qualified Zipline 3.2.0 daily fixture for a ten-share
order filled as five shares on each of two bars. It pays the one-dollar order
minimum once. The native per-fill alternative would pay two dollars.

<!-- ml4t-doc-test: order-context-fees -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode, ExecutionPrice
from ml4t.backtest.execution import SharedVolumeParticipationLimit

dates = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [6, 7, 8]]
prices = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 3,
                       "close": [104.0, 105.0, 106.0], "volume": [10.0] * 3})
class Buy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp == dates[0]:
            broker.submit_order("TEST", 10)
config = BacktestConfig(initial_cash=10000, calendar=None,
    execution_mode=ExecutionMode.NEXT_BAR,
    next_bar_execution_price=ExecutionPrice.CLOSE,
    commission_type=CommissionType.PER_ORDER_PER_SHARE,
    commission_per_share=0.01, commission_minimum=1.0)
result = Engine(DataFeed(prices_df=prices), Buy(), config,
    execution_limits=SharedVolumeParticipationLimit(max_participation=0.5)).run()
fills = result.to_fills_dataframe()
assert fills["quantity"].to_list() == [5.0, 5.0]
assert fills["timestamp"].to_list() == dates[1:]
assert fills["price"].to_list() == [105.0, 106.0]
assert fills["commission"].to_list() == [1.0, 0.0]
assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 9999.0, 10004.0]
print("Partial order fees and shared volume verified.")
```

### Route explicit fees by asset

`Engine`, `Engine.from_config`, `run_backtest` and `Broker.from_config` accept
`commission_model=` to override configured fee rates. `AssetCommission` routes
stable asset keys to native models, with an explicit default for other keys.
It passes the order's committed quantity and paid fees to models that implement
`quote(CommissionContext)`; other models receive the existing `calculate`
interface. Estimates use copies, so they cannot advance an actual model's
in-memory fill state.

Use `PerOrderCommission(per_share=..., per_trade=..., minimum=...)` for a
per-contract charge, a one-time exchange charge and an order minimum. The unit
charge counts contracts. Only the optional `rate` term uses multiplied
notional. `PerOrderCommission(per_trade=...)` also expresses a fixed futures
trade fee. There is no additional per-contract fee formula.

<!-- ml4t-doc-test: native-asset-commission -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
from ml4t.backtest.execution import VolumeParticipationLimit
from ml4t.backtest.models import AssetCommission, PerOrderCommission, PerOrderPerShareCommission

prices = pl.DataFrame([
    {"timestamp": datetime(2025, 3, day, 21, tzinfo=UTC), "asset": asset,
     "close": 100., "volume": 2.}
    for day in [3, 4, 5, 6] for asset in ["A", "ES"]
])
fees = AssetCommission({"ES": PerOrderCommission(per_share=2., per_trade=3., minimum=8.)},
                        default=PerOrderPerShareCommission(per_share=.2, minimum=1.))

class Buy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp.day == 3:
            broker.submit_order("A", 3)
            broker.submit_order("ES", 3)

result = Engine(DataFeed(prices_df=prices), Buy(),
                BacktestConfig(calendar=None, initial_cash=100000), commission_model=fees,
                execution_limits=VolumeParticipationLimit(.5),
                contract_specs={"ES": ContractSpec("ES", AssetClass.FUTURE, multiplier=50.)}).run()
fills = result.to_fills_dataframe()
assert fills.filter(pl.col("asset") == "ES")["commission"].to_list() == [8., 0., 1.]
assert fills.filter(pl.col("asset") == "A")["commission"].to_list() == [1., 0., 0.]
assert result.to_equity_dataframe()["equity"][-1] == 99990.
print("Per-asset fees retain committed partial fills and contract units.")
```

The futures minimum is paid on the first fill; cumulative charges reach 9 on
the third fill. A multiplier of 50 does not multiply those unit fees. The
published Zipline Reloaded 3.2.0 mixed-asset fixture checks this sequence and
the fixed futures trade fee against its actual engine, with an explicit
one-unit-per-bar caller slippage model. It does not establish default futures
slippage or settlement behavior. Native signed-notional cash remains unchanged;
the example's final cash is 84690, whereas Zipline's variation-cash account
reports 99690 with the same NAV. An explicit variation-cash account method
remains separate work.

A supplied Python model is not reconstructed from `BacktestConfig` serialization.
Retain its parameters with the run inputs and supply it again when rerunning a
simulation. Saved result artifacts retain the executed fees and account path.

### Slippage Models

`Engine`, `Engine.from_config`, `run_backtest` and `Broker.from_config` accept
an explicit `slippage_model=`. `AssetSlippage` selects native models by stable
key, with a required default. Its optional `price_sources` mapping sets the
execution basis for ordinary market and conditional orders. Explicit MOC close
orders and typed opening child intents retain their declared close/open basis.
Marks continue to use the separate marking policy.

`AssetExecutionLimits` selects existing participation limits by stable key.
Shared capacity remains owned by the native order state for each asset/bar;
separate orders cannot reuse it. Quote estimates copy only the selected model
and do not advance its actual fill state.

<!-- ml4t-doc-test: native-asset-execution -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode, ExecutionPrice
from ml4t.backtest.execution import AssetExecutionLimits, SharedVolumeParticipationLimit
from ml4t.backtest.models import AssetSlippage, FixedSlippage, PercentageSlippage

prices = pl.DataFrame([
    {"timestamp": datetime(2025, 3, day, 21, tzinfo=UTC), "asset": asset,
     "close": 100., "high": 120., "low": 100., "volume": 4.}
    for day in [3, 4, 5, 6] for asset in ["A", "ES"]
])
class Buy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp.day == 3:
            broker.submit_order("A", 3)
            broker.submit_order("ES", 3)

result = Engine(DataFeed(prices_df=prices), Buy(), BacktestConfig(
    calendar=None, execution_mode=ExecutionMode.NEXT_BAR,
    next_bar_execution_price=ExecutionPrice.CLOSE),
    slippage_model=AssetSlippage({"ES": FixedSlippage(.5)},
        default=PercentageSlippage(.01), price_sources={"ES": ExecutionPrice.MID}),
    execution_limits=AssetExecutionLimits({"ES": SharedVolumeParticipationLimit(.25)},
        default=SharedVolumeParticipationLimit(.5)),
    contract_specs={"ES": ContractSpec("ES", AssetClass.FUTURE, multiplier=50.)}).run()
fills = result.to_fills_dataframe()
assert fills.filter(pl.col("asset") == "A")["price"].to_list() == [101., 101.]
assert fills.filter(pl.col("asset") == "ES")["price"].to_list() == [110.5] * 3
assert result.to_portfolio_state_dataframe()["cash"][-1] == 83122.
assert result.to_equity_dataframe()["equity"][-1] == 98422.
print("Distinct asset execution bases, costs and volume caps verified.")
```

The futures fills use the observed high/low midpoint 110 plus a 0.5 adverse
cost; the equity uses close 100 plus 1%. These deliberately different sources
are recorded on the fills. The future is marked at 100 with multiplier 50,
so its marked loss includes the 10.5-point difference. This example selects
native signed-notional accounting and does not claim a variation-cash result.
Retain supplied model parameters alongside the inputs when rerunning; Python
model objects are not reconstructed by `BacktestConfig` serialization.

| Type | Calculation | Config |
|------|------------|--------|
| `NONE` | No slippage | |
| `PERCENTAGE` | % of price | `slippage_rate=0.001` (0.1%) |
| `FIXED` | Fixed $ per share | `slippage_fixed=0.01` |
| `VOLUME_BASED` | Linear size-to-volume impact | `slippage_rate=0.1` (impact coefficient) |
| `CUMULATIVE_VOLUME_SHARE` | Quadratic committed asset/bar participation | `slippage_rate=0.1` (impact coefficient) |

Slippage models remain separate from quote-side execution:

- `QUOTE_SIDE` crosses the observed spread using bid/ask quotes
- slippage adds an extra synthetic execution penalty on top of the chosen source

This lets you model spread and market impact separately.

Stop orders can have additional slippage via `stop_slippage_rate`:

```python
config = BacktestConfig(
    slippage_rate=0.001,        # 0.1% for market orders
    stop_slippage_rate=0.001,   # Additional 0.1% for stop exits
)
```

## Settlement

The `settlement_delay` parameter delays when sale proceeds become spendable. The
unit is processed bars, not calendar or business days:

```python
config = BacktestConfig(
    settlement_delay=2,                    # Proceeds held for two processed bars
    settlement_reduces_buying_power=True,  # Unsettled cash not spendable
)
```

Use this as an explicit simulation assumption. A value of two does not by itself
represent any market's legal settlement calendar because skipped sessions and bar
frequency change the elapsed time.

## Cash Management

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `initial_cash` | 100,000 | Starting cash |
| `cash_buffer_pct` | 0.0 | Reserve this % of cash (never invest it) |
| `reject_on_insufficient_cash` | True | Reject orders that exceed buying power |
| `skip_cash_validation` | False | Bypass gatekeeper entirely (Zipline-style) |
| `buying_power_reservation` | False | Reserve cash at submission time (LEAN-style) |

## Position Sizing

```python
from ml4t.backtest.config import ShareType

# Allow fractional shares (crypto, some brokers)
config = BacktestConfig(share_type=ShareType.FRACTIONAL)

# Round down to whole shares (most equity brokers)
config = BacktestConfig(share_type=ShareType.INTEGER)
```

## In the book

Chapter 16, Section 16.3, [Engine divergence anatomy](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/07_engine_divergence_anatomy.ipynb) isolates fill timing and ordering differences. This page specifies the current engine behavior behind those comparisons.

## Next Steps

- [Book Guide](../book-guide/index.md) -- chapter and case-study map for execution workflows
- [Configuration](configuration.md) -- complete reference for all 40+ parameters
- [Profiles](profiles.md) -- pre-configured settings for each framework
- [Risk Management](risk-management.md) -- position rules and portfolio limits
