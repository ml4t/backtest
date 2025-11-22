# Code Review Request: ml4t.backtest Engine

## What We're Building

An event-driven backtesting engine for quantitative trading strategies with:
- Institutional-grade execution fidelity
- Point-in-time correctness (no look-ahead bias)
- Configurable behavior to match other frameworks (VectorBT, Backtrader, Zipline)
- Support for ML-driven signals

## What to Review

Please review the implementation for:

### 1. **Core Execution Logic Correctness**
- `src/ml4t/backtest/broker.py` - Order execution, position tracking, fill simulation
- `src/ml4t/backtest/engine.py` - Event loop orchestration
- Are fills calculated correctly within OHLC bounds?
- Is position tracking consistent across buy/sell operations?

### 2. **Accounting & Constraints**
- `src/ml4t/backtest/accounting/` - Account policies (cash vs margin)
- `src/ml4t/backtest/accounting/gatekeeper.py` - Order validation
- Are buying power calculations correct for cash accounts?
- Are margin requirements properly enforced?

### 3. **Data Flow**
- `src/ml4t/backtest/datafeed.py` - Price + signal data iteration
- `src/ml4t/backtest/types.py` - Core data structures
- Is the strategy receiving correct OHLCV + signals at each timestamp?

### 4. **Configuration System**
- `src/ml4t/backtest/config.py` - Centralized configuration
- `src/ml4t/backtest/models.py` - Commission/slippage models
- Are the preset configurations reasonable defaults?

### 5. **API Design**
- `src/ml4t/backtest/strategy.py` - Strategy interface
- `src/ml4t/backtest/__init__.py` - Public API exports
- Is the strategy interface intuitive for users?
- Are the right things exported at the package level?

## Key Use Cases to Consider

1. **Single-asset, long-only strategy with ML signals**
   ```python
   class MLStrategy(Strategy):
       def on_data(self, timestamp, data, context, broker):
           signal = data["AAPL"]["signals"]["ml_score"]
           if signal > 0.7 and broker.get_position("AAPL") is None:
               broker.submit_order("AAPL", 100)
   ```

2. **Multi-asset with position sizing**
   - Equal-weighted across assets
   - Cash constraint enforcement

3. **Event-driven entries and exits**
   - Stop-loss and take-profit
   - Trailing stops

## Known Limitations

- No bracket orders (OCO, OTO) yet
- No multi-leg strategies (pairs, spreads)
- Execution assumes daily bars (no intraday simulation yet)

## What We DON'T Want

- Over-engineering or premature abstraction
- Excessive error handling for impossible states
- Documentation for documentation's sake

## Files Included

```
src/ml4t/backtest/
├── __init__.py          - Package exports
├── broker.py            - Core execution
├── config.py            - Configuration
├── datafeed.py          - Data iteration
├── engine.py            - Event loop
├── models.py            - Commission/slippage
├── strategy.py          - Strategy interface
├── types.py             - Core types
└── accounting/
    ├── account.py       - Account state
    ├── gatekeeper.py    - Order validation
    ├── models.py        - Position model
    └── policy.py        - Account policies

tests/
├── test_core.py         - Engine tests
└── accounting/          - Accounting tests
```

## Output Expected

Please provide:
1. **Bugs found** - Critical issues that would cause incorrect results
2. **Design concerns** - Architectural issues or code smells
3. **Improvement suggestions** - Nice-to-haves for later
4. **Questions** - Anything unclear about intent or design
