# Analysis Examples: Backtest → Diagnostic Integration

This directory contains examples demonstrating the integration between `ml4t.backtest` (execution) and `ml4t.diagnostic` (analysis).

## Examples Overview

| Example | Description | ML | Key Features |
|---------|-------------|-----|--------------|
| [01_single_asset_trade_stats.py](01_single_asset_trade_stats.py) | Simple momentum strategy on SPY | No | `BacktestAnalyzer`, `TradeStatistics` |
| [02_multi_asset_portfolio.py](02_multi_asset_portfolio.py) | Cross-asset momentum portfolio | No | Per-asset breakdown, temporal analysis |
| [03_ml_linear_regression.py](03_ml_linear_regression.py) | Linear regression return prediction | Yes | Walk-forward training, feature importance |
| [04_ml_gradient_boosting.py](04_ml_gradient_boosting.py) | Gradient boosting classification | Yes | Rich features, permutation importance |
| [05_benchmark_comparison.py](05_benchmark_comparison.py) | Strategy vs benchmark workflow | No | Statistical significance, full metrics |

## Running Examples

All examples use real ETF data from `~/ml4t/data/etfs/`. Run from the backtest directory:

```bash
cd /home/stefan/ml4t/software/backtest
uv run python examples/analysis/01_single_asset_trade_stats.py
```

## Key Concepts

### The Adapter Pattern

The `ml4t.backtest.analysis` module bridges backtest output to diagnostic analysis:

```python
from ml4t.backtest.analysis import BacktestAnalyzer, to_trade_records

# After running backtest
result = engine.run()

# Create analyzer
analyzer = BacktestAnalyzer(engine)

# Get trade statistics (no diagnostic dependency)
stats = analyzer.trade_statistics()
print(f"Win rate: {stats.win_rate:.2%}")

# Convert trades for diagnostic library
records = analyzer.get_trade_records()

# Now use with ml4t.diagnostic
from ml4t.diagnostic.evaluation import TradeAnalysis
diagnostic = TradeAnalysis(records)
```

### Available Metrics

The `TradeStatistics` class provides:
- Win rate, profit factor, payoff ratio
- Average P&L, expectancy
- Max winner/loser
- Holding period analysis
- Commission and slippage totals

### ML Strategy Analysis

For ML strategies, examples show:
1. Walk-forward (rolling) model training
2. Feature importance tracking
3. Prediction confidence analysis
4. SHAP integration patterns

### Statistical Testing

Example 05 demonstrates:
- Sharpe ratio comparison
- Statistical significance testing
- Sortino, Calmar ratios
- Drawdown analysis

## Integration with Diagnostic Library

For advanced analysis beyond what the backtest module provides:

```python
from ml4t.diagnostic.evaluation import (
    TradeAnalysis,
    TradeShapAnalyzer,
)
from ml4t.diagnostic.evaluation.stats import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

# SHAP-based error analysis
shap_analyzer = TradeShapAnalyzer(model, features, shap_values)
patterns = shap_analyzer.explain_worst_trades(worst_20)

# Statistical corrections for multiple testing
dsr = deflated_sharpe_ratio(observed_sr, n_trials=10, returns=returns)
```

## Data Requirements

Examples expect ETF data in Hive-partitioned format:
```
~/ml4t/data/etfs/ohlcv_1d/
├── ticker=SPY/data.parquet
├── ticker=TLT/data.parquet
├── ticker=GLD/data.parquet
└── ...
```

Columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
