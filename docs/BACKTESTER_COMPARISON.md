# Backtester Comparison Framework

## Overview

QuantLab includes integration with multiple established backtesting frameworks to:
1. Validate our qengine implementation against industry standards
2. Compare performance and results across different approaches
3. Enable migration paths from existing systems
4. Benchmark ML strategy performance

## Installed Backtesters

### 1. VectorBT (v0.27.1)
- **Type**: Vectorized backtesting
- **Strengths**: Fast array operations, portfolio optimization
- **Use Case**: Parameter sweeps, optimization studies
- **Documentation**: https://vectorbt.dev/

### 2. Zipline-Reloaded (v3.1.1)
- **Type**: Event-driven, Quantopian's legacy
- **Strengths**: Battle-tested, extensive ecosystem
- **Use Case**: Traditional algorithmic strategies
- **Documentation**: https://zipline.ml4trading.io/

### 3. Backtrader (v1.9.78.123)
- **Type**: Event-driven, Python-native
- **Strengths**: Flexible, extensive indicators
- **Use Case**: Technical analysis strategies
- **Documentation**: https://www.backtrader.com/

### 4. VectorBT Pro (Commercial - Optional)
```bash
# Install separately if you have a license:
pip install vectorbtpro
```
- **Type**: GPU-accelerated vectorized
- **Strengths**: Massive parallelization, cloud-ready
- **Use Case**: Large-scale optimization

### 5. NautilusTrader (Optional)
```bash
# Has specific requirements - install if needed:
pip install nautilus-trader
```
- **Type**: High-performance Rust/Python hybrid
- **Strengths**: Ultra-low latency, production-ready
- **Use Case**: HFT, market making
- **Note**: Requires Rust toolchain

## Installation

All backtesters (except commercial/optional ones) are installed via:
```bash
source .venv/bin/activate
uv pip install -e ".[backtest]"
```

## Comparison Testing Structure

```
integration_tests/
├── backtester_comparison/
│   ├── strategies/
│   │   ├── ml_strategy.py         # Common ML strategy
│   │   ├── mean_reversion.py      # Simple benchmark
│   │   └── trend_following.py     # Classic momentum
│   ├── adapters/
│   │   ├── qengine_adapter.py     # Our engine
│   │   ├── vectorbt_adapter.py    # VectorBT wrapper
│   │   ├── zipline_adapter.py     # Zipline wrapper
│   │   └── backtrader_adapter.py  # Backtrader wrapper
│   ├── benchmarks/
│   │   ├── performance_metrics.py # Speed comparisons
│   │   └── result_validation.py   # Result consistency
│   └── test_comparison.py         # Main comparison tests
```

## Usage Example

```python
from integration_tests.backtester_comparison import run_comparison

# Define common strategy
strategy = MLStrategy(
    features=qfeatures_pipeline.transform(data),
    model=qeval_validated_model
)

# Run on all backtesters
results = run_comparison(
    strategy=strategy,
    data=historical_data,
    engines=['qengine', 'vectorbt', 'zipline', 'backtrader']
)

# Compare results
print(results.performance_summary())
print(results.consistency_report())
print(results.speed_benchmarks())
```

## Key Metrics for Comparison

1. **Performance Metrics**
   - Total Returns
   - Sharpe Ratio
   - Maximum Drawdown
   - Win Rate

2. **Execution Differences**
   - Fill assumptions
   - Slippage models
   - Commission structures

3. **Speed Benchmarks**
   - Backtest runtime
   - Memory usage
   - Optimization time

4. **ML-Specific Features**
   - Point-in-time data handling
   - Feature pipeline integration
   - Model prediction timing

## Expected Differences

Different backtesters may produce slightly different results due to:
- Order execution models
- Bar timestamp interpretation
- Position sizing algorithms
- Corporate action handling

Our comparison framework documents and explains these differences.

## Next Steps

1. Implement common strategy interfaces
2. Create data adapters for each backtester
3. Build comparison test suite
4. Document migration guides from each system
