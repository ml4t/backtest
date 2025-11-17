# Cross-Framework Validation Architecture

## Purpose

Prove that **ml4t.backtest produces identical results to other frameworks** when given:
- Same signals (pre-calculated true/false)
- Same data (price/volume)
- Same execution rules

**What we validate:**
- ✅ Same trades (timestamp, ticker, side, quantity, price)
- ✅ Same positions over time
- ✅ Same P&L calculations
- ✅ Same commission/slippage application

**What we DON'T validate:**
- ❌ Strategy logic (signals are pre-computed)
- ❌ Indicator calculations (not relevant)
- ❌ Performance metrics (Sharpe, etc.) - can differ by implementation

---

## Architecture Overview

```
ml4t-backtest/
├── validation/                          # Separate from tests/
│   ├── README.md                        # "How to run validation"
│   ├── __init__.py
│   │
│   ├── data/                            # Canonical datasets
│   │   ├── load_canonical.py           # Data loaders
│   │   ├── crypto_daily.pkl            # From ../../../projects/crypto_futures
│   │   └── equities_daily.pkl          # From ../../../projects/wikipedia
│   │
│   ├── signals/                         # Pre-calculated signals
│   │   ├── generate_signals.py         # Signal generation scripts
│   │   ├── simple_ma_single.pkl        # Single asset, daily, SMA crossover
│   │   ├── simple_ma_multi.pkl         # Multi asset, daily, SMA crossover
│   │   └── momentum_minute.pkl         # Single asset, minute, momentum
│   │
│   ├── frameworks/                      # Framework adapters
│   │   ├── base.py                     # Base adapter interface
│   │   ├── ml4t_adapter.py             # ml4t.backtest
│   │   ├── backtrader_adapter.py       # Backtrader
│   │   ├── vectorbt_adapter.py         # VectorBT (open source)
│   │   └── zipline_adapter.py          # Zipline (with bundle setup)
│   │
│   ├── runners/                         # Execution scripts
│   │   ├── run_single_asset.py         # Single asset validation
│   │   ├── run_multi_asset.py          # Multi asset validation
│   │   └── run_all.py                  # Full validation suite
│   │
│   ├── comparison/                      # Result analysis
│   │   ├── trade_matcher.py            # Match trades across frameworks
│   │   ├── position_matcher.py         # Match positions over time
│   │   ├── pnl_matcher.py              # Match P&L calculations
│   │   └── report.py                   # Generate comparison report
│   │
│   └── results/                         # Output (gitignored)
│       ├── single_asset_daily.json
│       ├── multi_asset_daily.json
│       └── validation_report.md
```

---

## Data Sources

Use **existing canonical datasets** from `../../../projects/`:

### 1. Crypto Daily (from crypto_futures project)
- **Path**: `../../../projects/crypto_futures/data/`
- **Period**: 2020-2024
- **Tickers**: BTC, ETH, SOL (3 assets)
- **Frequency**: Daily
- **Use case**: Multi-asset daily validation

### 2. Equities Daily (from Wikipedia/Quandl)
- **Path**: `../../../projects/wikipedia/` (if exists, or create from Quandl)
- **Period**: 2010-2020
- **Tickers**: SPY, QQQ, IWM (3 ETFs)
- **Frequency**: Daily
- **Use case**: Equities multi-asset validation

### 3. Single Asset Minute (generate if needed)
- **Ticker**: SPY
- **Period**: 2023 (1 year)
- **Frequency**: Minute bars
- **Use case**: High-frequency validation

**Key principle**: Use small, manageable datasets. We're validating **correctness**, not scalability.

---

## Signal Generation

All frameworks use **identical pre-calculated signals**:

```python
# signals/generate_signals.py

import pandas as pd
import numpy as np

def generate_sma_crossover(prices: pd.DataFrame, fast=10, slow=20):
    """
    Generate simple SMA crossover signals.

    Returns:
        DataFrame with columns: ['entry', 'exit']
        - entry: True when fast SMA crosses above slow SMA
        - exit: True when fast SMA crosses below slow SMA
    """
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()

    # Entry: fast crosses above slow
    entry = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))

    # Exit: fast crosses below slow
    exit = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

    return pd.DataFrame({'entry': entry, 'exit': exit})
```

**Signal fixtures saved as pickles:**
- `simple_ma_single.pkl` - SPY, daily, SMA(10, 20)
- `simple_ma_multi.pkl` - [SPY, QQQ, IWM], daily, SMA(10, 20) per asset
- `momentum_minute.pkl` - SPY, minute, momentum breakout

---

## Adapter Interface

All frameworks implement same interface:

```python
# frameworks/base.py

from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class Trade:
    """Standardized trade record."""
    timestamp: datetime
    ticker: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0

@dataclass
class Position:
    """Standardized position snapshot."""
    timestamp: datetime
    ticker: str
    quantity: float
    avg_cost: float
    market_value: float

@dataclass
class BacktestResult:
    """Standardized backtest output."""
    framework: str
    trades: List[Trade]
    positions: List[Position]  # Snapshots over time
    final_portfolio_value: float
    total_pnl: float
    total_commission: float
    execution_time: float

class FrameworkAdapter:
    """Base class for framework adapters."""

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        initial_cash: float = 100_000,
        commission_rate: float = 0.001
    ) -> BacktestResult:
        """
        Run backtest with pre-calculated signals.

        Args:
            data: OHLCV data (index=timestamp, columns=[open,high,low,close,volume])
            signals: Entry/exit signals (index=timestamp, columns=[entry, exit])
            initial_cash: Starting cash
            commission_rate: Commission as fraction (0.001 = 0.1%)

        Returns:
            BacktestResult with standardized trade/position records
        """
        raise NotImplementedError
```

---

## Implementation Example: ml4t.backtest Adapter

```python
# frameworks/ml4t_adapter.py

from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.data.feed import DataFeed
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.execution.commission import PercentageCommission
from ml4t.backtest.core.event import MarketEvent
from .base import FrameworkAdapter, BacktestResult, Trade, Position

class SignalStrategy(Strategy):
    """Strategy that executes pre-calculated signals."""

    def __init__(self, signals: pd.DataFrame):
        super().__init__()
        self.signals = signals

    def on_event(self, event):
        if isinstance(event, MarketEvent):
            self.on_market_event(event)

    def on_market_event(self, event: MarketEvent):
        timestamp = event.timestamp

        # Check if we have a signal for this timestamp
        if timestamp not in self.signals.index:
            return

        signal_row = self.signals.loc[timestamp]

        # Execute entry
        if signal_row['entry']:
            # Buy with all available cash
            self.order_percent(event.asset_id, 1.0, event.close)

        # Execute exit
        elif signal_row['exit']:
            # Close position
            self.close_position(event.asset_id, event.close)

class ML4TAdapter(FrameworkAdapter):
    """Adapter for ml4t.backtest framework."""

    def run(self, data, signals, initial_cash=100_000, commission_rate=0.001):
        # Create data feed
        feed = DataFeedFromDataFrame(data, asset_id='TEST')

        # Create broker
        broker = SimulationBroker(
            initial_cash=initial_cash,
            commission_model=PercentageCommission(rate=commission_rate)
        )

        # Create strategy
        strategy = SignalStrategy(signals)

        # Run backtest
        engine = BacktestEngine(broker=broker, strategy=strategy, data_feed=feed)
        start_time = time.time()
        engine.run()
        execution_time = time.time() - start_time

        # Extract trades
        trades = self._extract_trades(broker)

        # Extract positions
        positions = self._extract_positions(broker)

        return BacktestResult(
            framework='ml4t.backtest',
            trades=trades,
            positions=positions,
            final_portfolio_value=broker.portfolio.equity,
            total_pnl=broker.portfolio.equity - initial_cash,
            total_commission=broker.portfolio.total_commission,
            execution_time=execution_time
        )
```

---

## Validation Process

### 1. Generate Signals (one-time setup)

```bash
cd validation/signals
python generate_signals.py
# Creates: simple_ma_single.pkl, simple_ma_multi.pkl, etc.
```

### 2. Run Single-Asset Validation

```bash
cd validation/runners
python run_single_asset.py --signal simple_ma_single --data crypto_daily
```

**Output:**
```
Running single-asset validation: simple_ma_single on crypto_daily
================================================================

Framework: ml4t.backtest
  ✓ Completed in 0.15s
  Trades: 42
  Final Value: $125,432.15

Framework: Backtrader
  ✓ Completed in 0.08s
  Trades: 42
  Final Value: $125,432.15

Framework: VectorBT
  ✓ Completed in 0.03s
  Trades: 42
  Final Value: $125,432.15

Framework: Zipline
  ✓ Completed in 1.2s
  Trades: 42
  Final Value: $125,432.15

Comparison:
  ✅ All frameworks produced 42 trades
  ✅ Trade timestamps match 100%
  ✅ Trade prices match within $0.01
  ✅ Final P&L matches within $0.10

VALIDATION PASSED ✅
```

### 3. Run Multi-Asset Validation

```bash
python run_multi_asset.py --signal simple_ma_multi --data crypto_daily
```

### 4. Generate Report

```bash
python run_all.py --output results/validation_report.md
```

---

## Trade Matching Logic

```python
# comparison/trade_matcher.py

def match_trades(trades_a: List[Trade], trades_b: List[Trade],
                 tolerance=0.01) -> Dict:
    """
    Match trades between two frameworks.

    Args:
        trades_a: Trades from framework A
        trades_b: Trades from framework B
        tolerance: Price tolerance in dollars

    Returns:
        {
            'matched': [(trade_a, trade_b), ...],
            'unmatched_a': [trade_a, ...],
            'unmatched_b': [trade_b, ...],
            'price_differences': [diff, ...],
            'max_price_diff': float
        }
    """
    matched = []
    unmatched_a = []
    unmatched_b = list(trades_b)

    for trade_a in trades_a:
        # Find matching trade in framework B
        match = None
        for i, trade_b in enumerate(unmatched_b):
            if (trade_a.timestamp == trade_b.timestamp and
                trade_a.ticker == trade_b.ticker and
                trade_a.side == trade_b.side and
                abs(trade_a.price - trade_b.price) <= tolerance):
                match = trade_b
                unmatched_b.pop(i)
                break

        if match:
            matched.append((trade_a, match))
        else:
            unmatched_a.append(trade_a)

    # Calculate price differences
    price_diffs = [abs(a.price - b.price) for a, b in matched]

    return {
        'matched': matched,
        'unmatched_a': unmatched_a,
        'unmatched_b': unmatched_b,
        'price_differences': price_diffs,
        'max_price_diff': max(price_diffs) if price_diffs else 0.0,
        'match_rate': len(matched) / max(len(trades_a), len(trades_b))
    }
```

---

## Separate from Main Tests

**Why this is NOT in `tests/`:**

1. **Heavy setup**: Zipline bundles, multiple frameworks, large data
2. **Not unit tests**: This is integration/validation, not functionality testing
3. **Infrequent runs**: Only needed during development or before releases
4. **Optional dependencies**: Requires backtrader, vectorbt, zipline installed
5. **Documentation value**: Results are publishable as proof of correctness

**Main test suite (`tests/`) focuses on:**
- Unit tests: Individual component correctness
- Integration tests: ml4t.backtest internal components
- Fast execution: All tests run in < 1 minute

**Validation suite (`validation/`) focuses on:**
- Cross-framework equivalence
- Publication-ready validation results
- Can be run separately: `python validation/run_all.py`

---

## Deployment Strategy

### During Development
```bash
# Activate ml4t-backtest environment
cd validation
python run_all.py
# Review results/validation_report.md
```

### Before Release
```bash
# Run full validation on clean environment
./validation/run_full_validation.sh
# Generates: VALIDATION_RESULTS_v0.1.0.md
# Include in package documentation
```

### CI/CD (Optional)
```yaml
# .github/workflows/validation.yml
name: Cross-Framework Validation
on:
  release:
    types: [published]

jobs:
  validate:
    steps:
      - Setup all frameworks
      - Run validation suite
      - Upload results as release asset
```

---

## Success Criteria

**Validation passes if:**
1. ✅ Trade count matches across all frameworks
2. ✅ Trade timestamps match exactly
3. ✅ Trade prices match within $0.01 (or 0.01%)
4. ✅ Final portfolio value matches within $1.00
5. ✅ Position quantities match within 0.01 shares

**If validation fails:**
- Identify divergence point (which trade differs)
- Debug framework-specific execution logic
- Fix and re-validate

---

## Timeline

**Phase 1** (2 hours): Setup structure
- Create validation/ directory
- Implement base adapter interface
- Load canonical data from projects/

**Phase 2** (3 hours): Generate signals
- SMA crossover (single asset, daily)
- SMA crossover (multi asset, daily)
- Momentum (single asset, minute)

**Phase 3** (4 hours): Implement adapters
- ml4t.backtest adapter
- Backtrader adapter
- VectorBT adapter
- Zipline adapter (with bundle)

**Phase 4** (2 hours): Comparison logic
- Trade matcher
- Position matcher
- Report generator

**Phase 5** (1 hour): Documentation
- README with usage instructions
- Example validation report

**Total: ~12 hours to complete validation infrastructure**

---

## Benefits

1. **Proof of correctness**: Can show ml4t.backtest matches industry standards
2. **Regression testing**: Detect if changes break compatibility
3. **User confidence**: Published validation results build trust
4. **Development tool**: Quickly verify new features match other frameworks
5. **Clean separation**: Heavy validation separate from fast unit tests

---

*This architecture provides a simple, pragmatic approach to cross-framework validation without burdening the main test suite.*
