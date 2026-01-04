# ml4t-backtest Agent Reference

## Purpose
Event-driven backtesting engine with point-in-time correctness, validated against VectorBT Pro.

## Installation
```bash
pip install ml4t-backtest
```

## Quick Start
```python
from ml4t.backtest import Engine, Strategy, BacktestConfig

class MyStrategy(Strategy):
    def on_bar(self, bar):
        if self.position == 0:
            self.buy(size=100)

config = BacktestConfig(initial_cash=100_000)
engine = Engine(data, MyStrategy(), config)
result = engine.run()
print(result.metrics)
```

## Core API

### Engine(data, strategy, config) -> Engine
Main backtest runner.
- data: DataFrame with OHLCV columns
- strategy: Strategy instance
- config: BacktestConfig
- returns: Engine instance

```python
engine = Engine(data, strategy, config)
result = engine.run()
```

### Strategy (base class)
Inherit to implement trading logic.

Key methods to override:
- `on_bar(bar)`: Called for each price bar
- `on_fill(fill)`: Called when order is filled

Key methods to call:
- `buy(size, price=None)`: Submit buy order
- `sell(size, price=None)`: Submit sell order
- `close()`: Close current position
- `position`: Current position size (int)
- `equity`: Current equity (float)

### BacktestConfig
```python
config = BacktestConfig(
    initial_cash=100_000,        # Starting capital
    commission=0.001,            # 0.1% per trade
    slippage=0.0005,             # 0.05% slippage
    margin_ratio=1.0,            # Cash account (no margin)
)
```

### BacktestResult
```python
result = engine.run()

# Key attributes
result.equity_curve      # DataFrame: datetime, equity
result.trades           # List of closed trades
result.metrics          # Dict: sharpe, return, drawdown, etc.
result.final_equity     # Float
result.total_return     # Float (percentage)

# Methods
result.to_tearsheet()   # Generate HTML report
result.to_vectorbt()    # Convert to VectorBT format
```

## Order Types
- `OrderType.MARKET`: Execute at next bar open
- `OrderType.LIMIT`: Execute if price crosses limit
- `OrderType.STOP`: Trigger market order at stop price
- `OrderType.STOP_LIMIT`: Trigger limit order at stop price

## Account Policies

### Cash Account (Default)
```python
config = BacktestConfig(margin_ratio=1.0)  # No leverage
```

### Margin Account
```python
config = BacktestConfig(margin_ratio=0.5)  # 2x leverage
```

## Common Patterns

### Mean Reversion Strategy
```python
class MeanReversion(Strategy):
    def __init__(self, lookback=20, z_threshold=2.0):
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.prices = []
    
    def on_bar(self, bar):
        self.prices.append(bar.close)
        if len(self.prices) < self.lookback:
            return
        
        mean = np.mean(self.prices[-self.lookback:])
        std = np.std(self.prices[-self.lookback:])
        z_score = (bar.close - mean) / std
        
        if z_score < -self.z_threshold and self.position <= 0:
            self.buy(size=100)
        elif z_score > self.z_threshold and self.position >= 0:
            self.sell(size=100)
```

### Momentum Strategy with Risk Management
```python
class MomentumWithRisk(Strategy):
    def __init__(self, fast=10, slow=30, risk_per_trade=0.02):
        self.fast = fast
        self.slow = slow
        self.risk_per_trade = risk_per_trade
        
    def on_bar(self, bar):
        fast_ma = bar.close_ma(self.fast)
        slow_ma = bar.close_ma(self.slow)
        
        if fast_ma > slow_ma and self.position == 0:
            # Position size based on risk
            stop_distance = bar.atr(14) * 2
            size = int(self.equity * self.risk_per_trade / stop_distance)
            self.buy(size=size)
            self.set_stop(bar.close - stop_distance)
```

## Configuration Presets
```python
from ml4t.backtest.presets import crypto_futures, us_equities

config = crypto_futures()   # Binance-like fees
config = us_equities()      # IB-like fees
```

## Error Handling
| Error | Cause | Fix |
|-------|-------|-----|
| InsufficientFunds | Order exceeds available cash | Reduce size or use margin |
| InvalidOrder | Negative size or invalid price | Validate before submit |
| NoDataError | Data range too short | Provide more history |

## What NOT To Do
- DON'T use future prices (look-ahead bias)
- DON'T ignore transaction costs
- DON'T skip the VectorBT validation
- DON'T assume fills at exact limit price

## VectorBT Validation
Results are validated to match VectorBT Pro exactly:
```python
from ml4t.backtest.validation import validate_against_vectorbt

is_valid = validate_against_vectorbt(result, vbt_result)
assert is_valid, "Results don't match VectorBT"
```

## Integration with ML4T Libraries
```python
from ml4t.data import DataManager
from ml4t.engineer import compute_features
from ml4t.diagnostic import Evaluator
from ml4t.diagnostic.splitters import CombinatorialPurgedCV
from ml4t.backtest import Engine, Strategy

# 1. Load data
data = DataManager().fetch("SPY", "2020-01-01", "2023-12-31")

# 2. Generate signals with features
features = compute_features(data, ["rsi_14", "macd", "bbands_20_2"])
signals = model.predict(features)

# 3. Backtest
class SignalStrategy(Strategy):
    def __init__(self, signals):
        self.signals = signals
        
    def on_bar(self, bar):
        signal = self.signals.get(bar.datetime)
        if signal > 0.5 and self.position == 0:
            self.buy(size=100)
        elif signal < -0.5 and self.position > 0:
            self.close()

result = Engine(data, SignalStrategy(signals)).run()

# 4. Validate with cross-validation
cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=2)
evaluation = Evaluator(splitter=cv).evaluate_backtest(result)
```
