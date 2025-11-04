# Backtester Comparison Tests

This directory contains comparison tests between qengine and established backtesting frameworks.

## Structure

```
comparison/
├── strategies/           # Common strategies for all backtesters
│   ├── __init__.py
│   ├── base.py          # Abstract strategy interface
│   ├── ml_strategy.py   # ML-based strategy
│   └── benchmarks.py    # Simple benchmark strategies
├── adapters/            # Wrappers for each backtester
│   ├── __init__.py
│   ├── base.py          # Common adapter interface
│   ├── qengine.py       # Our engine adapter
│   ├── vectorbt.py      # VectorBT/Pro adapter
│   ├── zipline.py       # Zipline adapter
│   └── backtrader.py    # Backtrader adapter
├── metrics/             # Comparison metrics
│   ├── __init__.py
│   ├── performance.py   # Return metrics
│   └── consistency.py   # Result validation
└── test_comparison.py   # Main comparison tests
```

## Available Backtesters

1. **qengine** - Our implementation
2. **VectorBT Pro** - GPU-accelerated vectorized backtesting
3. **VectorBT** - Open-source vectorized backtesting
4. **Zipline-Reloaded** - Quantopian's legacy, event-driven
5. **Backtrader** - Popular Python backtester

## Running Comparison Tests

```bash
# From monorepo root
make test-comparison

# Or directly
pytest qengine/tests/comparison/ -v

# Run specific comparison
pytest qengine/tests/comparison/test_comparison.py::test_ml_strategy_comparison -v
```

## Key Comparisons

### 1. ML Strategy Performance
- Tests qfeatures → qeval → backtester pipeline
- Compares prediction timing and execution
- Validates point-in-time correctness

### 2. Execution Models
- Order fill assumptions
- Slippage and commission handling
- Position sizing differences

### 3. Performance Benchmarks
- Backtest speed (rows/second)
- Memory usage
- Optimization capabilities

### 4. Result Consistency
- Total returns alignment
- Trade timing differences
- Risk metric calculations

## Expected Differences

Small differences between backtesters are normal due to:
- Timestamp interpretation (open vs close of bar)
- Order execution timing
- Rounding in position sizes
- Corporate action handling

Our tests document and quantify these differences.
