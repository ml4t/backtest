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
    min_trade_value=100,        # Skip trades smaller than $100
    min_weight_change=0.01,     # Skip changes smaller than 1%
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

## Next Steps

- [Strategies](strategies.md) -- strategy patterns and templates
- [Configuration](configuration.md) -- rebalance-related config parameters
- [Results & Analysis](results.md) -- analyze portfolio outcomes
