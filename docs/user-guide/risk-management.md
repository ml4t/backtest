# Risk Management

ml4t-backtest has two levels of risk management: **position rules** (per-position exits) and **portfolio limits** (portfolio-wide constraints). Position rules are the primary tool -- they automatically evaluate on every bar and generate exit orders when triggered.

## Position Rules

### StopLoss

Exit when loss exceeds a threshold:

```python
from ml4t.backtest import StopLoss

rule = StopLoss(pct=0.05)  # Exit at -5% from entry
```

- **Long positions**: triggers if `bar_low <= entry_price * (1 - pct)`
- **Short positions**: triggers if `bar_high >= entry_price * (1 + pct)`
- Gap handling: if bar opens beyond stop, fills at open price

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

Exit when loss exceeds N standard deviations of recent volatility:

```python
from ml4t.backtest.risk.position import VolatilityStop

rule = VolatilityStop(
    n_std=2.0,       # 2 standard deviations
    lookback=20,     # 20-bar window for volatility
)
```

### TighteningTrailingStop

Trail that tightens as profit increases:

```python
from ml4t.backtest.risk.position import TighteningTrailingStop

rule = TighteningTrailingStop(
    thresholds=[
        (0.05, 0.03),  # At +5% profit, trail 3%
        (0.10, 0.02),  # At +10% profit, trail 2%
        (0.20, 0.01),  # At +20% profit, trail 1%
    ],
)
```

### ScaledExit

Take partial profits at predefined levels:

```python
from ml4t.backtest.risk.position import ScaledExit

rule = ScaledExit(
    levels=[
        (0.10, 0.5),  # At +10%, exit 50% of position
        (0.20, 0.5),  # At +20%, exit remaining 50%
    ],
)
```

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

# Only exit if profitable AND held long enough
rule = AllOf([
    TakeProfit(pct=0.0),    # Must be profitable
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
    AllOf([TakeProfit(pct=0.0), TimeExit(max_bars=5)]),  # Profitable + held 5 bars
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
    max_drawdown=0.20,       # Halt at -20% drawdown
    warn_threshold=0.15,     # Warn at -15%
)
```

### MaxPositionsLimit

```python
from ml4t.backtest.risk.portfolio.limits import MaxPositionsLimit

limit = MaxPositionsLimit(max_positions=10)
```

### MaxExposureLimit

```python
from ml4t.backtest.risk.portfolio.limits import MaxExposureLimit

limit = MaxExposureLimit(max_exposure=2.0)  # Max 200% gross exposure
```

### DailyLossLimit

```python
from ml4t.backtest.risk.portfolio.limits import DailyLossLimit

limit = DailyLossLimit(max_daily_loss=0.03)  # Halt at -3% daily loss
```

### GrossExposureLimit / NetExposureLimit

```python
from ml4t.backtest.risk.portfolio.limits import GrossExposureLimit, NetExposureLimit

gross = GrossExposureLimit(max_gross=1.5)  # Max 150% gross
net = NetExposureLimit(min_net=-0.2, max_net=1.2)  # Net between -20% and 120%
```

### VaRLimit / CVaRLimit

```python
from ml4t.backtest.risk.portfolio.limits import VaRLimit, CVaRLimit

var_limit = VaRLimit(max_var=0.05, confidence=0.95)
cvar_limit = CVaRLimit(max_cvar=0.08, confidence=0.95)
```

### BetaLimit

```python
from ml4t.backtest.risk.portfolio.limits import BetaLimit

beta_limit = BetaLimit(max_beta=1.5)
```

### SectorExposureLimit / FactorExposureLimit

```python
from ml4t.backtest.risk.portfolio.limits import SectorExposureLimit, FactorExposureLimit

sector = SectorExposureLimit(max_sector_weight=0.30)
factor = FactorExposureLimit(max_factor_exposure=0.50)
```

## Limit Actions

Each limit check returns a `LimitResult` with an action:

| Action | Meaning |
|--------|---------|
| `none` | No breach |
| `warn` | Log warning, continue trading |
| `reduce` | Reduce position sizes by a percentage |
| `halt` | Stop opening new positions |

## See It in Action

The [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) book demonstrates risk management in Ch19 case studies:

- **ETFs** — RuleChain with StopLoss + TrailingStop on multi-asset ETF portfolios
- **FX Pairs** — StopLoss + TakeProfit + TrailingStop for currency strategies
- **CME Futures** — Risk rules with ContractSpec and per-contract commission
- **US Equities** — MaxDrawdownLimit and DailyLossLimit portfolio protection

The case studies show progressive complexity: basic stop-loss → trailing stops → rule chains → portfolio limits.

## Next Steps

- [Execution Semantics](execution-semantics.md) -- stop fill modes and trailing stop timing
- [Strategies](strategies.md) -- integrating risk rules into strategies
- [Configuration](configuration.md) -- stop-related config parameters
