# Risk Management

The [risk and state tutorial](../tutorials/risk-and-state.md) runs a stop, two
portfolio reductions, and the recovery between breaches from one declared price panel.

ml4t-backtest has two levels of risk management: **position rules** (per-position exits) and **portfolio limits** (portfolio-wide constraints). Position rules are the primary tool -- they automatically evaluate on every bar and generate exit orders when triggered.

Rule snippets below assume a strategy or broker with current positions and prices. The linked risk tutorial runs the full event sequence and inspects reduction fills.

## Validate submissions and accounts

These validators are development-source features, outside the preserved 0.1.13
release candidate. They use the existing Broker, order book and account ledger.
They raise on a violation; they do not request a portfolio liquidation.

Register `OrderConstraints` through `broker.register_trading_control(...)`.
Validation runs before an order consumes an identifier, count or cash, and also
applies to updates of a pending order. A failed strategy callback restores order
counts with its other broker mutations.

<!-- ml4t-doc-test: trading-account-controls -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode
from ml4t.backtest.controls import AccountConstraints, OrderConstraints, TradingControlViolation

dates = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [2, 3]]
prices = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 2,
                       "close": [100.0, 100.0]})

class Controlled(Strategy):
    def on_start(self, broker):
        broker.register_trading_control(OrderConstraints(
            max_order_quantity=5, max_position_quantity=5,
            max_orders_per_session=1, long_only=True,
        ))
        broker.register_account_control(AccountConstraints(max_leverage=1))

    def on_data(self, timestamp, data, context, broker):
        if timestamp == dates[0]:
            try:
                broker.submit_order("TEST", 6)
            except TradingControlViolation:
                pass
            assert broker.orders == []
            assert broker.submit_order("TEST", 3).order_id == "ORD-1"

result = Engine(DataFeed(prices_df=prices,
                        feed_spec={"timestamp_semantics": "bar_close"}),
                Controlled(), BacktestConfig(initial_cash=10000,
                                              execution_mode=ExecutionMode.NEXT_BAR)).run()
assert result.to_fills_dataframe()["quantity"].to_list() == [3.0]
print("Submission and observed-account controls verified.")
```

Position controls include same-direction unfilled quantities by default. Opposing
pending orders do not reduce that worst-case exposure, and an unfilled buy cannot
authorize a short sale. Choose `include_pending=False` for the legacy rule that
tests only held quantity plus the new order. Notional limits include the actual
instrument multiplier; `notional_basis="quoted_units"` explicitly selects the
legacy price-times-quantity convention.

`max_orders_per_session` requires a calendar and follows its exchange session,
including the upcoming session for a scheduled pre-open callback.
`max_orders_per_day` counts the delivered timestamp's civil date. Failed validation
does not consume counts; a submitted order counts even if later execution rejects
it. Updating or cancelling an order does not create another submission.
`blocked_assets` and the optional pure `restriction(request)` callable restrict
stable asset IDs at the order's actual timestamp. Custom trading validators expose
`validate(request)` and inspect the immutable `OrderControlContext`.

Account rules inspect one native marked account snapshot per price event. The
default `boundary="after_fills"` includes all that event's fills. Choose
`boundary="after_callbacks"` to validate after strategy/scheduled callbacks,
before deferred same-bar or market-on-close fills. Immediate fills made inside a
callback are already part of that snapshot. `min_leverage` requires a datetime
`deadline` and applies strictly after it. Custom account validators expose
`validate(account)` over immutable account values. A breach aborts the run without
undoing completed callbacks; use portfolio limits below for halt or liquidation.
`broker.validate_account_controls()` can also validate all registered rules
immediately.

For a partially filled order, `broker.update_order(order_id, quantity=...)`
changes the positive, finite remaining quantity. Earlier fills remain recorded.
The update cannot bypass trading controls or leave the old executable remainder
in the partial-fill book.

## Position Rules

### StopLoss

Exit when loss exceeds a threshold:

```python
from ml4t.backtest import StopLoss

rule = StopLoss(pct=0.05)  # Exit at -5% from entry
```

- **Long positions**: triggers if `bar_low <= entry_price * (1 - pct)`
- **Short positions**: triggers if `bar_high >= entry_price * (1 + pct)`
- With the default `STOP_PRICE` fill mode, a gap through the stop fills at the bar open; other modes use their configured fill rule

### TakeProfit

Exit when profit reaches a target:

```python
from ml4t.backtest import TakeProfit

rule = TakeProfit(pct=0.15)  # Exit at +15% from entry
```

- **Long positions**: triggers if `bar_high >= entry_price * (1 + pct)`
- **Short positions**: triggers if `bar_low <= entry_price * (1 - pct)`

### TrailingStop

Exit when price retraces from the high water mark:

```python
from ml4t.backtest import TrailingStop

rule = TrailingStop(pct=0.03)  # Exit at 3% retrace from peak
```

- Tracks the highest price since entry (longs) or lowest price since entry (shorts)
- Triggers when price retraces by `pct` from the water mark
- Water mark behavior is configurable via `trail_hwm_source`, `initial_hwm_source`, and `trail_stop_timing` in BacktestConfig

See [Execution Semantics](execution-semantics.md#trailing-stop-mechanics) for the full details on trailing stop timing modes.

### TimeExit

Exit after holding for a maximum number of bars:

```python
from ml4t.backtest.risk.position import TimeExit

rule = TimeExit(max_bars=20)  # Exit after 20 bars
```

### VolatilityStop

Exit when price moves against a position by a multiple of Average True Range (ATR).
After a position exists, supply a positive `atr` value through
`broker.update_position_context(asset, {"atr": current_atr})`. Without it, the rule holds:

<!-- ml4t-doc-test: risk-volatility-stop -->
```python
from ml4t.backtest.risk.position import VolatilityStop

rule = VolatilityStop(
    multiplier=2.0,  # Stop distance is twice the supplied ATR
)
```

### TighteningTrailingStop

Trail that tightens as profit increases:

<!-- ml4t-doc-test: risk-tightening-trailing-stop -->
```python
from ml4t.backtest.risk.position import TighteningTrailingStop

rule = TighteningTrailingStop(
    schedule=[
        (0.00, 0.05),  # Before +5% profit, trail 5%
        (0.05, 0.03),  # At +5% profit, trail 3%
        (0.10, 0.02),  # At +10% profit, trail 2%
        (0.20, 0.01),  # At +20% profit, trail 1%
    ],
)
```

### ScaledExit

Take partial profits at predefined levels:

<!-- ml4t-doc-test: risk-scaled-exit -->
```python
from ml4t.backtest.risk.position import ScaledExit

rule = ScaledExit(
    targets=[
        (0.10, 0.5),  # At +10%, exit 50% of current position
        (0.20, 1.0),  # At +20%, exit the remaining position
    ],
)
```

`ScaledExit` tracks triggered targets on the rule instance. Use a separate instance for each
position or reset it when a position closes.

### SignalExit

Exit based on a signal value in the position's context:

```python
from ml4t.backtest.risk.position import SignalExit

rule = SignalExit(threshold=0.3)  # Exit when signal drops below 0.3
```

## Composing Rules

### RuleChain (First Trigger Wins)

The most common pattern -- rules evaluate in order, first non-HOLD action triggers:

```python
from ml4t.backtest import RuleChain, StopLoss, TakeProfit, TrailingStop

rules = RuleChain([
    StopLoss(pct=0.05),        # Highest priority: hard stop at -5%
    TakeProfit(pct=0.20),      # Take profit at +20%
    TrailingStop(pct=0.03),    # Trail 3% from peak
])
```

### AllOf (All Must Agree)

Exit only when multiple conditions are true simultaneously:

```python
from ml4t.backtest.risk.position import AllOf, TakeProfit, TimeExit

# Exit at or above breakeven after at least five bars
rule = AllOf([
    TakeProfit(pct=0.0),    # At or above breakeven
    TimeExit(max_bars=5),   # Must have held 5+ bars
])
```

### AnyOf (Any Trigger Wins)

Semantically equivalent to RuleChain, but named for clarity when composing:

```python
from ml4t.backtest.risk.position import AnyOf

rule = AnyOf([
    StopLoss(pct=0.05),
    SignalExit(threshold=0.3),
])
```

### Nested Composition

Combine composition patterns for complex logic:

```python
rules = RuleChain([
    StopLoss(pct=0.08),                    # Hard stop always applies
    AllOf([TakeProfit(pct=0.0), TimeExit(max_bars=5)]),  # Breakeven + held 5 bars
    TrailingStop(pct=0.03),                # Trail from peak
    TimeExit(max_bars=60),                 # Max hold 60 bars
])
```

## Setting Rules

### Global Rules

Apply to all positions:

```python
class MyStrategy(Strategy):
    def on_start(self, broker):
        broker.set_position_rules(RuleChain([
            StopLoss(pct=0.05),
            TrailingStop(pct=0.03),
        ]))
```

### Per-Asset Rules

Override rules for specific assets:

```python
def on_start(self, broker):
    # Global default
    broker.set_position_rules(StopLoss(pct=0.05))

    # Override for volatile assets
    broker.set_position_rules(
        RuleChain([StopLoss(pct=0.10), TrailingStop(pct=0.05)]),
        asset="TSLA",
    )
```

Per-asset rules take precedence over global rules for that asset.

## Portfolio Limits

Portfolio limits operate at the portfolio level, not per-position. They check aggregate metrics (drawdown, exposure, position count) and can warn, reduce positions, or halt trading.

Import from `ml4t.backtest.risk.portfolio.limits`:

### MaxDrawdownLimit

```python
from ml4t.backtest.risk.portfolio.limits import MaxDrawdownLimit

limit = MaxDrawdownLimit(
    max_drawdown=0.20,       # Liquidate at -20% drawdown
    warn_threshold=0.15,     # Warn at -15%
)
```

### MaxPositionsLimit

```python
from ml4t.backtest.risk.portfolio.limits import MaxPositionsLimit

limit = MaxPositionsLimit(max_positions=10)
```

### MaxExposureLimit

<!-- ml4t-doc-test: risk-max-exposure -->
```python
from ml4t.backtest.risk.portfolio.limits import MaxExposureLimit

limit = MaxExposureLimit(max_exposure_pct=0.10)  # Warn above 10% in one asset
```

### DailyLossLimit

<!-- ml4t-doc-test: risk-daily-loss -->
```python
from ml4t.backtest.risk.portfolio.limits import DailyLossLimit

limit = DailyLossLimit(max_daily_loss_pct=0.03)  # Liquidate above 3% daily loss
```

### GrossExposureLimit / NetExposureLimit

<!-- ml4t-doc-test: risk-gross-net -->
```python
from ml4t.backtest.risk.portfolio.limits import GrossExposureLimit, NetExposureLimit

gross = GrossExposureLimit(max_gross_exposure=1.5)  # Halt above 150% gross
net = NetExposureLimit(min_net_exposure=-0.2, max_net_exposure=1.2)  # Warn outside range
```

### VaRLimit / CVaRLimit

These checks require at least `lookback_days` of portfolio returns in the risk manager's
`context["historical_returns"]`. Without that input they report no breach.

<!-- ml4t-doc-test: risk-var-cvar -->
```python
from ml4t.backtest.risk.portfolio.limits import VaRLimit, CVaRLimit

var_limit = VaRLimit(threshold=0.05, confidence_level=0.95)
cvar_limit = CVaRLimit(threshold=0.08, confidence_level=0.95)
```

### BetaLimit

```python
from ml4t.backtest.risk.portfolio.limits import BetaLimit

beta_limit = BetaLimit(max_beta=1.5)
```

### SectorExposureLimit / FactorExposureLimit

Sector checks require an asset-to-sector mapping in `context["asset_sectors"]`.
Factor checks require an asset-to-loading mapping in `context["factor_loadings"]`;
the example names that factor momentum. Pass these mappings as the `context` argument to
`RiskManager.update(...)`. Without them the checks report no breach.

<!-- ml4t-doc-test: risk-sector-factor -->
```python
from ml4t.backtest.risk.portfolio.limits import SectorExposureLimit, FactorExposureLimit

sector = SectorExposureLimit(max_sector_exposure=0.30)
factor = FactorExposureLimit(factor_name="momentum", max_exposure=0.50)
```

## Limit Actions

Each limit check returns a `LimitResult` with an action:

| Action | Meaning |
|--------|---------|
| `none` | No breach |
| `warn` | Log warning, continue trading |
| `reduce` | Reduce open positions by the configured fraction; currently supported by `MaxDrawdownLimit` |
| `halt` | Stop opening new positions |
| `liquidate` | Flatten open positions and stop new trading |

For a drawdown reduction, set both `action="reduce"` and an explicit
`reduction_pct` in (0, 1]. The manager cancels pending orders and submits one
risk-tagged reduction order per open position. It acts once while that limit
remains breached and can act again after the breach clears. Other built-in
limits reject `action="reduce"` at construction until they have a defined
reduction policy.

For example, `MaxDrawdownLimit(max_drawdown=0.10, action="reduce",
reduction_pct=0.50)` cuts each open position by half after a 10% drawdown.

Pass the broker into `RiskManager.update(...)` so reduction and liquidation
actions are applied through normal order execution:

```python
results = risk_manager.update(
    equity=broker.get_account_value(),
    positions={asset: pos.market_value for asset, pos in broker.positions.items()},
    timestamp=timestamp,
    broker=broker,
)

if risk_manager.is_halted:
    return
```

## In the book

Chapter 19, Section 19.4, [ml4t-backtest risk demo](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/10_ml4t_backtest_risk_demo.ipynb) applies position rules and portfolio limits. The [exit strategies notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/02_exit_strategies.ipynb) compares stop choices.

## Next Steps

- [Book Guide](../book-guide/index.md) -- risk-management chapter and case-study mapping
- [Execution Semantics](execution-semantics.md) -- stop fill modes and trailing stop timing
- [Strategies](strategies.md) -- integrating risk rules into strategies
- [Configuration](configuration.md) -- stop-related config parameters
