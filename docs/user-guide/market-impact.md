# Market Impact & Execution Costs

Realistic backtesting requires modeling the costs of executing trades. ml4t-backtest provides three layers of cost modeling: commission, slippage, and market impact.

## Cost Layers

| Layer | What It Models | Config |
|-------|---------------|--------|
| **Commission** | Broker fees | `commission_type`, `commission_rate` |
| **Slippage** | Bid-ask spread crossing | `slippage_type`, `slippage_rate` |
| **Market impact** | Price movement from your order | `market_impact_model=` kwarg |

Commission and slippage are configured via BacktestConfig. Market impact is an optional model passed to the Engine.

## Commission Models

### Percentage (Default)

```python
from ml4t.backtest import BacktestConfig

config = BacktestConfig(
    commission_rate=0.001,  # 10 bps per trade
)
```

### Per-Share

```python
from ml4t.backtest import BacktestConfig, CommissionType

config = BacktestConfig(
    commission_type=CommissionType.PER_SHARE,
    commission_per_share=0.005,   # $0.005 per share
    commission_minimum=1.0,       # $1 minimum per trade
)
```

### Per-Contract (Futures)

```python
config = BacktestConfig(
    commission_type=CommissionType.PER_CONTRACT,
    commission_per_share=2.50,  # $2.50 per contract
)
```

`PER_CONTRACT` is an alias for `PER_SHARE` — same math, clearer intent for futures.

### Custom Models

For volume-tiered or combined commission structures, use model objects:

```python
from ml4t.backtest.models import TieredCommission, CombinedCommission

# Volume-tiered (Interactive Brokers style)
tiered = TieredCommission(tiers=[
    (300, 0.0035),    # First 300 shares: $0.0035/share
    (3000, 0.0020),   # 301-3000: $0.0020/share
    (float('inf'), 0.0015),  # 3001+: $0.0015/share
])

# Combined (base + percentage)
combined = CombinedCommission(
    fixed=1.0,        # $1 base
    per_share=0.005,  # Plus $0.005/share
)
```

## Slippage Models

Slippage models the bid-ask spread you cross when executing. A buy order fills slightly above the mid-price; a sell order fills slightly below.

### Percentage (Default)

```python
config = BacktestConfig(
    slippage_rate=0.001,        # 10 bps for market orders
    stop_slippage_rate=0.001,   # Additional 10 bps for stop exits
)
```

Stop exits can have additional slippage because stops trigger during fast markets.

### Fixed

```python
from ml4t.backtest.config import SlippageType

config = BacktestConfig(
    slippage_type=SlippageType.FIXED,
    slippage_fixed=0.01,  # $0.01 per share
)
```

## Market Impact Models

Market impact captures the price movement caused by your order itself — large orders move the market. This is the most important cost for institutional-size strategies.

Import from `ml4t.backtest.execution`:

```python
from ml4t.backtest.execution import LinearImpact, SquareRootImpact, NoImpact
```

### No Impact (Default)

```python
engine = Engine(feed, strategy, config)
# Equivalent to: market_impact_model=NoImpact()
```

### Linear Impact

Price impact proportional to order size relative to bar volume:

$$\text{impact} = \eta \times \frac{Q}{V}$$

where $Q$ = order quantity, $V$ = bar volume, $\eta$ = impact coefficient.

```python
from ml4t.backtest.execution import LinearImpact

engine = Engine(
    feed, strategy, config,
    market_impact_model=LinearImpact(eta=0.1),
)
```

An order that is 10% of bar volume with `eta=0.1` moves the fill price by 1%.

### Square-Root Impact

The standard institutional model — impact scales with the square root of participation rate:

$$\text{impact} = \eta \times \sigma \times \sqrt{\frac{Q}{V}}$$

where $\sigma$ = daily volatility, $\eta$ = impact coefficient.

```python
from ml4t.backtest.execution import SquareRootImpact

engine = Engine(
    feed, strategy, config,
    market_impact_model=SquareRootImpact(eta=0.5),
)
```

Square-root impact is the empirical consensus for equity markets (Almgren-Chriss, Barra).

### Volume Participation Limits

Prevent orders from consuming too much bar volume:

```python
from ml4t.backtest.execution import VolumeParticipationLimit

engine = Engine(
    feed, strategy, config,
    execution_limits=VolumeParticipationLimit(max_participation=0.10),
)
```

Orders exceeding 10% of bar volume are partially filled (the remainder stays pending).

## Cost Impact Analysis

To measure cost impact, run the same strategy with and without costs:

```python
# Full costs
config_real = BacktestConfig(
    commission_rate=0.002,
    slippage_rate=0.002,
)

# Zero costs
config_zero = BacktestConfig(
    commission_rate=0.0,
    slippage_rate=0.0,
)

result_real = Engine(feed, strategy, config_real).run()
result_zero = Engine(feed2, strategy2, config_zero).run()

cost_drag = result_zero.metrics['total_return_pct'] - result_real.metrics['total_return_pct']
print(f"Cost drag: {cost_drag:.2f}%")
```

## See It in Action

The [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) book demonstrates market impact in Ch18:

- **Cost notebooks** — LinearImpact and SquareRootImpact models applied to multi-asset portfolios
- **VolumeParticipationLimit** — preventing oversized orders in illiquid assets
- **Cost drag analysis** — comparing gross vs net returns across case studies

## Next Steps

- [Execution Semantics](execution-semantics.md) — fill timing, ordering, and stop modes
- [Configuration](configuration.md) — all commission and slippage parameters
- [Rebalancing](rebalancing.md) — how costs interact with weight-based rebalancing
