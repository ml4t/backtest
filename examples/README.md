# ml4t.backtest Examples

This directory contains working examples demonstrating portfolio optimization and analysis integration with ml4t.backtest.

## Quick Start

```bash
cd /path/to/backtest
source .venv/bin/activate

# Basic example (no dependencies)
python examples/01_basic_multi_asset.py

# Risk parity (no dependencies)
python examples/04_risk_parity.py

# Trade analysis examples
python examples/analysis/01_single_asset_trade_stats.py

# With riskfolio-lib
pip install riskfolio-lib
python examples/02_riskfolio_meanvariance.py

# With skfolio
pip install skfolio
python examples/03_skfolio_integration.py
```

## Example Categories

### Portfolio Optimization (`./`)
Examples 01-04 focus on portfolio construction and rebalancing.

### Analysis & Diagnostics (`./analysis/`)
Examples demonstrating backtest → diagnostic integration. See [analysis/README.md](analysis/README.md).

| Example | Description |
|---------|-------------|
| `analysis/01_single_asset_trade_stats.py` | Trade statistics (win rate, profit factor) |
| `analysis/02_multi_asset_portfolio.py` | Per-asset and temporal analysis |
| `analysis/03_ml_linear_regression.py` | ML strategy with feature importance |
| `analysis/04_ml_gradient_boosting.py` | Classification strategy with SHAP patterns |
| `analysis/05_benchmark_comparison.py` | Benchmark vs strategy statistical testing |

---

## Portfolio Optimization Examples

### 01_basic_multi_asset.py
**No external dependencies**

Basic multi-asset backtest with equal-weight rebalancing. Demonstrates:
- Multi-asset portfolio with 5 stocks
- Monthly rebalancing to equal weights
- Using `TargetWeightExecutor` for weight-to-order conversion

### 02_riskfolio_meanvariance.py
**Requires: `pip install riskfolio-lib`**

Mean-Variance optimization using [riskfolio-lib](https://riskfolio-lib.readthedocs.io/). Demonstrates:
- Rolling window mean-variance optimization
- Maximum Sharpe ratio portfolio
- Quarterly rebalancing
- Position constraints (max 40% per asset)

### 03_skfolio_integration.py
**Requires: `pip install skfolio`**

Portfolio optimization using [skfolio](https://skfolio.org/) (scikit-learn compatible). Demonstrates:
- Scikit-learn style fit-predict paradigm
- Multiple optimization models:
  - `MeanRisk`: Mean-variance optimization
  - `HierarchicalRiskParity`: HRP clustering-based allocation
  - `EqualWeighted`: Baseline
- Walk-forward optimization pattern

### 04_risk_parity.py
**No external dependencies**

Risk parity (inverse volatility) portfolio implemented from scratch. Demonstrates:
- Custom optimizer without external libraries
- Volatility-based weight calculation
- Integration with `TargetWeightExecutor`

## Key Pattern: TargetWeightExecutor

All examples use `TargetWeightExecutor` to convert portfolio weights to orders:

```python
from ml4t.backtest import TargetWeightExecutor, RebalanceConfig

executor = TargetWeightExecutor(
    config=RebalanceConfig(
        min_trade_value=500,      # Skip trades < $500
        min_weight_change=0.02,   # Skip if weight change < 2%
        allow_fractional=True,    # Allow fractional shares
        allow_short=False,        # Long-only
        max_single_weight=0.40,   # Max 40% per asset
    )
)

# In strategy.on_data():
target_weights = optimizer.get_weights(...)  # From any optimizer
orders = executor.execute(target_weights, data, broker)
```

## Integration Pattern

### Any Portfolio Optimizer → ml4t.backtest

```python
class OptimizedStrategy(Strategy):
    def __init__(self, optimizer, assets, lookback, rebalance_freq):
        self.optimizer = optimizer
        self.assets = assets
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.prices = {a: [] for a in assets}
        self.bar_count = 0

        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(min_trade_value=500)
        )

    def on_data(self, timestamp, data, context, broker):
        # Update price history
        for asset in self.assets:
            if asset in data:
                self.prices[asset].append(data[asset]['close'])

        self.bar_count += 1

        # Check if enough history
        if min(len(p) for p in self.prices.values()) < self.lookback:
            return

        # Rebalance check
        if self.bar_count % self.rebalance_freq != 0:
            return

        # Get returns for optimizer
        returns = self._compute_returns()

        # YOUR OPTIMIZER HERE
        weights = self.optimizer.optimize(returns)

        # Execute
        orders = self.executor.execute(weights, data, broker)
```

## Supported Optimizers

| Library | Pattern | Example |
|---------|---------|---------|
| **riskfolio-lib** | `port.optimization()` → dict | `02_riskfolio_meanvariance.py` |
| **skfolio** | `optimizer.fit(X)` → `weights_` | `03_skfolio_integration.py` |
| **PyPortfolioOpt** | `ef.max_sharpe()` → dict | Similar to riskfolio |
| **cvxpy** | Custom convex optimization | Use result directly |
| **Custom** | Any function returning weights | `04_risk_parity.py` |

## Performance Considerations

- **Rebalance frequency**: Monthly (21 bars) or quarterly (63 bars) is typical
- **Lookback window**: 126 bars (~6 months) for stable covariance estimates
- **Min trade thresholds**: Set `min_trade_value` to avoid churning on small changes
- **Weight thresholds**: Set `min_weight_change` to skip trivial rebalances

## Common Issues

### "Optimization failed"
- Check if returns have NaN values
- Ensure enough history (lookback period)
- Some optimizers fail on highly correlated assets

### Orders not executing
- Check `min_trade_value` threshold
- Check `min_weight_change` threshold
- Verify prices are in `data` dict

### Weight sum > 1.0
- `TargetWeightExecutor` automatically scales down
- Or set `max_single_weight` constraint
