# ml4t.backtest Validation - Implementation Guide

**Status**: Ready to Begin
**Approach**: Systematic, quality-first validation (no artificial deadlines)
**Goal**: Comprehensive validation before any release

---

## Phase 0: Infrastructure Setup

### Task 0.1: Framework Installation (Isolated Environments)

**Recommendation**: Separate virtual environments for cleaner dependency management

```bash
# Create isolated environments
cd ~/ml4t/backtest

# VectorBT Pro environment
python -m venv .venv-vectorbt
source .venv-vectorbt/bin/activate
cd resources/vectorbt.pro-main
pip install -e .
deactivate

# Zipline environment
python -m venv .venv-zipline
source .venv-zipline/bin/activate
pip install zipline-reloaded
deactivate

# Backtrader environment
python -m venv .venv-backtrader
source .venv-backtrader/bin/activate
pip install backtrader
deactivate

# ml4t.backtest environment (main development)
python -m venv .venv-ml4t.backtest
source .venv-ml4t.backtest/bin/activate
pip install -e .
# Install test dependencies
pip install pytest pandas polars numpy
deactivate
```

**Verification Script** (`scripts/verify_frameworks.py`):
```python
#!/usr/bin/env python
"""Verify all frameworks are properly installed."""

import sys
import subprocess

frameworks = [
    ('.venv-vectorbt', 'vectorbtpro'),
    ('.venv-zipline', 'zipline'),
    ('.venv-backtrader', 'backtrader'),
    ('.venv-ml4t.backtest', 'ml4t.backtest'),
]

for venv, package in frameworks:
    cmd = f"source {venv}/bin/activate && python -c 'import {package}; print({package}.__version__)'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {package}: {result.stdout.strip()}")
    else:
        print(f"❌ {package}: FAILED - {result.stderr}")
```

**Acceptance**: All 4 frameworks import successfully with version numbers displayed

---

### Task 0.2: Framework Adapter Implementation

**Extend existing base** (`tests/validation/frameworks/base.py`):

#### VectorBT Pro Adapter
```python
# tests/validation/frameworks/vectorbtpro_adapter.py

import subprocess
import json
from .base import BaseFrameworkAdapter, ValidationResult, TradeRecord

class VectorBTProAdapter(BaseFrameworkAdapter):
    """
    Adapter for VectorBT Pro using isolated environment.

    Note: Runs in separate process due to environment isolation.
    Uses quantgpt.chat for VectorBT Pro-specific questions.
    """

    def __init__(self):
        super().__init__(framework_name="VectorBT Pro")
        self.venv_path = ".venv-vectorbt"

    def run_backtest(self, data, strategy_params, initial_capital=10000):
        # Serialize inputs
        test_data = {
            'data': data.to_dict(),
            'strategy_params': strategy_params,
            'initial_capital': initial_capital
        }

        # Execute in isolated environment
        script = self._generate_vbt_script(test_data)
        result = self._run_in_venv(script)

        # Parse and return standardized results
        return self._parse_vbt_results(result)

    def _run_in_venv(self, script):
        """Execute Python script in VectorBT Pro environment."""
        cmd = f"source {self.venv_path}/bin/activate && python -c '{script}'"
        # Implementation details...
```

#### Zipline Adapter
```python
# tests/validation/frameworks/zipline_adapter.py

class ZiplineAdapter(BaseFrameworkAdapter):
    """
    Adapter for Zipline-Reloaded.

    Known issues from previous validation:
    - Complex bundle setup required
    - Timezone handling issues
    - Different results due to adjusted vs raw prices
    """

    def __init__(self):
        super().__init__(framework_name="Zipline-Reloaded")
        self.venv_path = ".venv-zipline"

    def run_backtest(self, data, strategy_params, initial_capital=10000):
        # Zipline-specific implementation
        # Handle bundle creation, data ingestion
        # Manage timezone complications
        pass
```

#### Backtrader Adapter
```python
# tests/validation/frameworks/backtrader_adapter.py

class BacktraderAdapter(BaseFrameworkAdapter):
    """
    Adapter for Backtrader.

    Known issues from previous validation:
    - Signal execution bug (missing trades)
    - Only use for basic Tier 1 validation
    """

    def __init__(self):
        super().__init__(framework_name="Backtrader")
        self.venv_path = ".venv-backtrader"

    def run_backtest(self, data, strategy_params, initial_capital=10000):
        # Backtrader-specific implementation
        # Known to have bugs - document carefully
        pass
```

**Acceptance**:
- All adapters implement `run_backtest()` correctly
- Return standardized `ValidationResult` objects
- Handle framework-specific quirks gracefully

---

### Task 0.3: Universal Data Loader

Create unified data loading that works across all frameworks:

```python
# tests/validation/data_loader.py

import polars as pl
import pandas as pd
from pathlib import Path

class UniversalDataLoader:
    """Load and convert data for any framework."""

    @staticmethod
    def load_daily_equities(symbol: str, start_date: str = None, end_date: str = None):
        """Load from ~/ml4t/projects/daily_us_equities/"""
        path = Path("~/ml4t/projects/daily_us_equities/equity_prices_enhanced_1962_2025.parquet").expanduser()

        # Load with Polars (fast)
        df = pl.read_parquet(path)

        # Filter by symbol and date
        if symbol:
            df = df.filter(pl.col('symbol') == symbol)
        if start_date:
            df = df.filter(pl.col('timestamp') >= start_date)
        if end_date:
            df = df.filter(pl.col('timestamp') <= end_date)

        # Convert to pandas (most frameworks expect this)
        return df.to_pandas().set_index('timestamp')

    @staticmethod
    def load_minute_bars(symbol: str, year: int):
        """Load from ~/ml4t/projects/nasdaq100_minute_bars/"""
        path = Path(f"~/ml4t/projects/nasdaq100_minute_bars/{year}.parquet").expanduser()
        # Implementation...

    @staticmethod
    def load_crypto_futures(symbol: str):
        """Load from ~/ml4t/projects/crypto_futures/data/futures/"""
        # Implementation...

    @staticmethod
    def to_framework_format(df: pd.DataFrame, framework: str):
        """Convert to framework-specific format."""
        if framework == 'vectorbt':
            # VectorBT expects specific column names
            return df.rename(columns={'Close': 'close', 'Volume': 'volume'})
        elif framework == 'zipline':
            # Zipline needs special preparation
            return df  # TODO: bundle creation
        elif framework == 'backtrader':
            # Backtrader format
            return df
        elif framework == 'ml4t.backtest':
            # ml4t.backtest uses Polars
            return pl.from_pandas(df)
```

**Acceptance**:
- Can load all data sources from ~/ml4t/projects/
- Converts formats correctly for each framework
- Handles missing data gracefully

---

### Task 0.4: Baseline Test - "Hello World" Backtest

**Simple MA Crossover on All Frameworks**:

```python
# tests/validation/test_00_baseline.py

import pytest
from validation.frameworks import ml4t.backtestAdapter, VectorBTProAdapter, ZiplineAdapter, BacktraderAdapter
from validation.data_loader import UniversalDataLoader

def test_baseline_all_frameworks():
    """Run simple MA crossover on all frameworks to verify setup."""

    # Load data
    data = UniversalDataLoader.load_daily_equities(
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2020-12-31'
    )

    # Strategy parameters
    strategy_params = {
        'name': 'MA Crossover',
        'short_window': 20,
        'long_window': 50
    }

    # Run on all frameworks
    adapters = [
        ml4t.backtestAdapter(),
        VectorBTProAdapter(),
        ZiplineAdapter(),
        BacktraderAdapter()
    ]

    results = {}
    for adapter in adapters:
        result = adapter.run_backtest(data, strategy_params, initial_capital=10000)
        results[adapter.framework_name] = result

        # Basic validation
        assert result.final_value > 0, f"{adapter.framework_name} failed"
        assert result.num_trades >= 0, f"{adapter.framework_name} invalid trade count"
        print(f"{adapter.framework_name}: ${result.final_value:,.2f} ({result.num_trades} trades)")

    # Compare results
    ml4t.backtest_value = results['ml4t.backtest'].final_value
    vbt_value = results['VectorBT Pro'].final_value

    # Allow 5% difference for baseline
    agreement = abs(ml4t.backtest_value - vbt_value) / ml4t.backtest_value * 100
    print(f"\nml4t.backtest vs VectorBT Pro agreement: {100-agreement:.2f}%")

    assert agreement < 5.0, f"Baseline test shows {agreement:.1f}% difference"
```

**Acceptance**:
- Test runs successfully on all 4 frameworks
- ml4t.backtest vs VectorBT Pro within 5% (for initial setup)
- All adapters return valid `ValidationResult` objects

---

## Phase 1: Tier 1 - Core Validation

**Goal**: Establish fundamental correctness (95%+ agreement with VectorBT Pro)

### Task 1.1: MA Crossover (Already Validated ✅)

**Document existing validation**:
- AAPL 2014-2015: 100% agreement ($1,507.06)
- 30-stock portfolio: 100% agreement with 5,000 trades

**Action**: Create standardized report template

### Task 1.2: RSI Mean Reversion Strategy

```python
# tests/validation/test_01_rsi_mean_reversion.py

def test_rsi_mean_reversion():
    """Validate RSI mean reversion strategy across frameworks."""

    # Load data
    data = UniversalDataLoader.load_daily_equities('SPY', '2020-01-01', '2022-12-31')

    # Strategy: Buy RSI < 30, Sell RSI > 70
    strategy_params = {
        'name': 'RSI Mean Reversion',
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
        'position_size': 1000  # Fixed shares
    }

    # Run validation
    results = run_cross_framework_validation(data, strategy_params)

    # Compare
    assert_agreement(results, threshold=0.95)
    generate_report(results, 'TIER1_RSI_MEAN_REVERSION.md')
```

**Acceptance**:
- 95%+ agreement between ml4t.backtest and VectorBT Pro
- Trade counts within ±5%
- Execution timing correct (no lookahead)

### Task 1.3: Bollinger Band Breakout

```python
def test_bollinger_breakout():
    """Bollinger breakout with stop-loss."""

    strategy_params = {
        'name': 'Bollinger Breakout',
        'period': 20,
        'std': 2.0,
        'stop_loss': 0.02,  # 2% stop
        'volume_filter': 1.5  # 1.5x average volume
    }

    # Test on multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    for symbol in symbols:
        data = UniversalDataLoader.load_daily_equities(symbol, '2021-01-01', '2022-12-31')
        results = run_cross_framework_validation(data, strategy_params)
        assert_agreement(results, threshold=0.95)
```

### Task 1.4: Multi-Indicator Combination

```python
def test_macd_rsi_combo():
    """MACD + RSI confirmation strategy."""

    strategy_params = {
        'name': 'MACD + RSI',
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_period': 14,
        'rsi_threshold': 50
    }

    # Entry: MACD crossover + RSI > 50
    # Exit: MACD crossunder OR RSI < 30
```

### Task 1.5: Multi-Asset Momentum Ranking

```python
def test_momentum_ranking():
    """Portfolio ranking strategy (already validated with 5,000 trades)."""

    # Load 30-stock universe
    symbols = ['AAPL', 'MSFT', 'GOOGL', ...]  # 30 stocks

    strategy_params = {
        'name': 'Momentum Ranking',
        'lookback': 20,
        'n_long': 5,   # Long top 5
        'n_short': 5,  # Short bottom 5
        'rebalance_freq': 5  # Every 5 days
    }
```

**Tier 1 Deliverable**: `TIER1_CORE_VALIDATION_REPORT.md`
- All 5 strategies validated
- Agreement percentages documented
- Known discrepancies explained
- Framework-specific notes

---

## Phase 2: Tier 2 - Advanced Execution

**Goal**: Validate order types and position sizing

### Task 2.1: Limit Orders

```python
def test_limit_orders():
    """Validate limit order execution and fill timing."""

    strategy_params = {
        'order_type': 'LIMIT',
        'limit_offset': -0.01,  # Buy 1% below market
        'timeout': 3  # Cancel after 3 bars
    }
```

### Task 2.2: Stop-Loss Orders

```python
def test_stop_loss():
    """Stop-loss triggering accuracy."""

    strategy_params = {
        'order_type': 'STOP',
        'stop_pct': 0.02,  # 2% stop
        'trigger_method': 'close'  # vs 'intrabar'
    }
```

### Task 2.3: Bracket Orders

```python
def test_bracket_orders():
    """Entry + stop + target in single bracket."""

    strategy_params = {
        'order_type': 'BRACKET',
        'stop_loss': 0.02,
        'take_profit': 0.05,
        'entry_type': 'LIMIT'
    }
```

### Task 2.4: Trailing Stops

```python
def test_trailing_stops():
    """Dynamic stop adjustment."""

    strategy_params = {
        'order_type': 'TRAILING_STOP',
        'trail_pct': 0.03,  # Trail 3% from peak
        'activation': 0.05  # Activate after 5% profit
    }
```

### Task 2.5: Minute Bar Strategies

```python
def test_minute_bar_execution():
    """Intraday execution on minute data."""

    data = UniversalDataLoader.load_minute_bars('SPY', 2022)

    strategy_params = {
        'name': 'Intraday Momentum',
        'frequency': 'minute',
        'ema_fast': 5,
        'ema_slow': 15
    }
```

**Tier 2 Deliverable**: `TIER2_EXECUTION_VALIDATION_REPORT.md`

---

## Phase 3: Tier 3 - ML Integration

**Goal**: Validate qfeatures → qeval → ml4t.backtest pipeline

### Task 3.1: qfeatures Technical Signals

```python
def test_qfeatures_technical():
    """Generate signals using qfeatures technical indicators."""

    from qfeatures import TechnicalFeatures

    # Load data
    data = UniversalDataLoader.load_daily_equities('AAPL', '2020-01-01', '2022-12-31')

    # Generate features with qfeatures
    tech = TechnicalFeatures(data)
    features = tech.compute(['rsi', 'macd', 'bollinger'])

    # Create signals
    signals = generate_signals_from_features(features)

    # Execute in all frameworks using SAME signals
    results = run_cross_framework_validation_with_signals(data, signals)
```

### Task 3.2: Binary Classification Signals

```python
def test_ml_binary_classifier():
    """ML model buy/sell predictions."""

    from qfeatures import FeatureEngineer
    from qeval import ModelValidator

    # Feature engineering
    engineer = FeatureEngineer(data)
    features = engineer.create_feature_set(['technical', 'momentum'])

    # Train model (using qeval)
    validator = ModelValidator(features, target='future_returns')
    model = validator.train_model('xgboost', params={'seed': 42})

    # Generate signals
    predictions = model.predict(features)
    signals = (predictions > 0.6)  # Buy when confidence > 0.6

    # Execute
    results = run_cross_framework_validation_with_signals(data, signals)
```

### Task 3.3: Return Forecasting for Sizing

```python
def test_ml_position_sizing():
    """ML return forecasts drive position sizing."""

    # Forecast returns
    forecasts = model.predict_returns(features)

    # Size positions based on forecast magnitude
    position_sizes = forecasts.clip(-1, 1) * max_position_size

    # Execute with dynamic sizing
```

### Task 3.4: Multi-Asset Ranking

```python
def test_ml_asset_ranking():
    """ML scores for portfolio construction."""

    # Score all assets
    scores = model.score_assets(features_multi_asset)

    # Rank and select
    long_positions = scores.nlargest(5)
    short_positions = scores.nsmallest(5)
```

### Task 3.5: Order Flow Microstructure

```python
def test_order_flow_signals():
    """SPY order flow features."""

    # Load order flow data
    flow_data = pd.read_parquet('~/ml4t/projects/spy_order_flow/spy_features.parquet')

    # Microstructure signals
    signals = (flow_data['imbalance'] > threshold)
```

**Tier 3 Deliverable**: `TIER3_ML_PIPELINE_VALIDATION_REPORT.md`

---

## Phase 4: Tier 4 - Performance & Edge Cases

### Performance Benchmarks

```python
def benchmark_single_asset_speed():
    """Measure events/sec, trades/sec."""
    # 10 years daily, simple strategy

def benchmark_multi_asset_scaling():
    """Test 10, 50, 100, 500 assets."""

def benchmark_hf_tick_data():
    """High-frequency tick processing."""

def benchmark_ml_overhead():
    """ML strategy vs simple strategy overhead."""
```

### Edge Case Tests

```python
def test_corporate_actions():
    """Dividends, splits, mergers."""

def test_extreme_markets():
    """2008 crash, 2020 COVID crash."""

def test_1000_asset_portfolio():
    """Scalability stress test."""
```

**Tier 4 Deliverable**: `TIER4_PERFORMANCE_AND_EDGE_CASES.md`

---

## Final Deliverable: Production Guide

**Consolidated Documentation**:
1. `VALIDATION_SUMMARY.md` - Executive summary
2. `FRAMEWORK_SELECTION_GUIDE.md` - When to use which framework
3. `KNOWN_LIMITATIONS.md` - Edge cases and workarounds
4. `PRODUCTION_READINESS_CHECKLIST.md` - Deployment guide

---

## Implementation Tracking

Use `.claude/planning/validation_progress.md` to track:

```markdown
## Phase 0: Setup
- [ ] Task 0.1: Install frameworks (separate venvs)
- [ ] Task 0.2: Implement adapters
- [ ] Task 0.3: Create data loaders
- [ ] Task 0.4: Run baseline test

## Phase 1: Tier 1 Core
- [✅] Task 1.1: MA Crossover (already done)
- [ ] Task 1.2: RSI Mean Reversion
- [ ] Task 1.3: Bollinger Breakout
- [ ] Task 1.4: MACD + RSI
- [ ] Task 1.5: Momentum Ranking

[Continue for all tasks...]
```

---

## Quality Over Speed

**No artificial deadlines.** Each phase completes when:
- ✅ All tests pass with 95%+ agreement
- ✅ Documentation complete and reviewed
- ✅ Edge cases identified and handled
- ✅ Code reviewed and refactored

**When in doubt**: Investigate thoroughly, document findings, ask for clarification.

This is about **correctness and confidence**, not speed.
