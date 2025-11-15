# Cross-Platform Backtest Validation Framework

**Version**: 1.0
**Date**: 2025-10-08
**Status**: ✅ Implementation Complete - Ready for Testing

## 🎯 Overview

Comprehensive validation framework for comparing ml4t.backtest against 4 other backtesting platforms:
- **ml4t.backtest** (our implementation)
- **VectorBT Pro** (commercial)
- **VectorBT Free** (open source)
- **Zipline-Reloaded** (Quantopian successor)
- **Backtrader** (popular Python framework)

### Key Features

✅ **Platform-Independent Signals**: Pure signal generation (no platform coupling)
✅ **Stop Loss / Take Profit**: Full support across all platforms
✅ **Trailing Stops**: Where supported by platforms
✅ **Trade-Level Comparison**: Detailed drilldown to individual trades
✅ **Multiple Strategies**: MA crossover, mean reversion, random signals
✅ **HTML Reports**: Visual comparison with difference highlighting
✅ **Real Data**: Uses `../projects/` datasets (63 years of daily US equities)

## 📁 Framework Architecture

```
tests/validation/
├── README.md                   # Quick start guide
├── FRAMEWORK_GUIDE.md          # This file (comprehensive documentation)
├── run_validation.py           # Main entry point (executable)
├── quick_test.py               # Component test script
│
├── signals/                    # Platform-independent signal generators
│   ├── __init__.py
│   ├── base.py                 # Signal and SignalGenerator abstractions
│   ├── ma_crossover.py         # Moving average crossover
│   ├── mean_reversion.py       # RSI-based mean reversion
│   └── random_signals.py       # Random signals (stress test)
│
├── adapters/                   # Platform-specific adapters
│   ├── __init__.py
│   ├── base.py                 # PlatformAdapter, Trade, BacktestResult
│   ├── ml4t.backtest_adapter.py      # ml4t.backtest integration
│   ├── vectorbt_adapter.py     # VectorBT Pro & Free
│   ├── zipline_adapter.py      # Zipline-reloaded
│   └── backtrader_adapter.py   # Backtrader
│
├── data/                       # Data loading utilities
│   ├── __init__.py
│   └── loader.py               # DataLoader for ../projects/
│
├── validators/                 # Result comparison and validation
│   ├── __init__.py
│   └── trade_validator.py      # TradeValidator, ValidationReport
│
├── test_cases/                 # Future: pytest test scenarios
│   └── (to be created)
│
└── results/                    # Validation outputs (gitignored)
    └── YYYY-MM-DD_HH-MM-SS/
        ├── ml4t.backtest_results.json
        ├── vectorbt_pro_results.json
        ├── zipline_results.json
        ├── backtrader_results.json
        └── validation_report.html
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From backtest/ directory
pip install polars pandas numpy

# Platform dependencies
pip install vectorbt              # Free version
pip install vectorbt-pro          # Commercial (if licensed)
pip install zipline-reloaded      # May have complex deps
pip install backtrader            # Lightweight

# ml4t.backtest (already in project)
# Uses local ml4t.backtest from src/
```

### 2. Run Basic Validation

```bash
cd tests/validation

# Test ml4t.backtest only (simplest)
python3 run_validation.py --strategy ma_cross --platforms ml4t.backtest

# Compare ml4t.backtest vs VectorBT Free
python3 run_validation.py --strategy ma_cross --platforms ml4t.backtest,vectorbt_free

# Compare all platforms
python3 run_validation.py --strategy ma_cross --platforms all

# Test all strategies
python3 run_validation.py --strategy all --platforms ml4t.backtest,vectorbt_free
```

### 3. View Results

Results saved to `results/YYYY-MM-DD_HH-MM-SS/`:
- `validation_report.html` - Visual comparison with trade drilldown
- `*_results.json` - Raw results from each platform

## 📊 Available Strategies

### 1. MA Crossover (Simple)
```bash
--strategy ma_cross
```
- Fast MA (10) crosses Slow MA (30)
- Fixed quantity (100 shares)
- No stop loss / take profit

### 2. MA Crossover with Risk Management
```bash
--strategy ma_cross_sl
```
- Same as above but with:
  - 5% stop loss
  - 10% take profit

### 3. Mean Reversion (RSI-based)
```bash
--strategy mean_reversion
```
- RSI(14) < 30 = Buy
- RSI(14) > 70 = Sell
- Exit when RSI ~50
- 5% stop loss, 10% take profit

### 4. Random Signals (Stress Test)
```bash
--strategy random
```
- Random entry/exit with 5% frequency
- Tests platform handling of arbitrary signals
- Fixed seed (42) for reproducibility
- Includes random stop loss / take profit

## 🔧 Advanced Usage

### Custom Date Range

```bash
python3 run_validation.py \
  --strategy ma_cross \
  --platforms ml4t.backtest,vectorbt_free \
  --start-date 2015-01-01 \
  --end-date 2020-12-31
```

### Multiple Symbols

```bash
python3 run_validation.py \
  --strategy ma_cross \
  --platforms ml4t.backtest \
  --symbols AAPL,GOOGL,MSFT
```

### Custom Capital and Commission

```bash
python3 run_validation.py \
  --strategy ma_cross \
  --platforms ml4t.backtest \
  --capital 1000000 \
  --commission 0.0005  # 5 basis points
```

## 📋 Validation Levels

The framework validates at multiple levels:

### Level 1: Signal Consistency
- ✅ All platforms receive identical signals
- ✅ Signal count matches across platforms
- **Expected**: 100% match

### Level 2: Order Generation
- ⚠️ Signals translated to equivalent orders
- ⚠️ Platform-specific order types may differ
- **Expected**: Semantically equivalent

### Level 3: Trade Execution
- ⚠️ Entry/exit times match (within tolerance)
- ⚠️ Fill prices match (within tolerance)
- **Tolerance**: ±0.1% for prices, ±1 bar for timing

### Level 4: P&L and Performance
- ⚠️ Trade-by-trade P&L
- ⚠️ Total P&L and metrics
- **Tolerance**: ±1% or documented differences

## 🎓 Adding New Strategies

### 1. Create Signal Generator

```python
# signals/my_strategy.py
from .base import Signal, SignalGenerator
import polars as pl

class MyStrategySignals(SignalGenerator):
    def __init__(self, param1: float = 1.0, name: str = "MyStrategy"):
        super().__init__(name)
        self.param1 = param1

    def generate_signals(self, data: pl.DataFrame) -> list[Signal]:
        self.validate_data(data)  # Checks required columns

        signals = []
        # Your logic here
        # Example:
        for row in data.iter_rows(named=True):
            if some_condition:
                signals.append(Signal(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    action='BUY',
                    quantity=100,
                    stop_loss=row['close'] * 0.95,  # Optional
                    take_profit=row['close'] * 1.10,  # Optional
                ))

        return signals
```

### 2. Register in run_validation.py

```python
# Add to get_available_strategies()
'my_strategy': lambda: MyStrategySignals(param1=2.0),
```

### 3. Run

```bash
python3 run_validation.py --strategy my_strategy --platforms ml4t.backtest
```

## 🐛 Troubleshooting

### Import Errors

```bash
# Check which modules are missing
python3 quick_test.py

# Install missing dependencies
pip install polars pandas numpy vectorbt backtrader
```

### VectorBT Pro Not Available

```bash
# Use free version instead
--platforms ml4t.backtest,vectorbt_free

# Or install pro (requires license)
pip install vectorbt-pro
```

### Zipline Bundle Errors

Zipline may require custom bundle setup. If errors occur:
- Framework will handle gracefully
- Results will show error in metadata
- Continue with other platforms

### No Data Found

```bash
# Verify ../projects/ exists and has data
ls ../projects/daily_us_equities/

# Framework expects:
../projects/daily_us_equities/equity_prices_enhanced_1962_2025.parquet
# OR
../projects/daily_us_equities/wiki_prices.parquet
```

## 📈 Interpreting Results

### HTML Report Sections

1. **Summary**
   - Trade count match: Should be identical
   - P&L comparison: Shows platform differences
   - Execution time: Performance comparison

2. **Trade-Level Differences**
   - Shows each discrepancy
   - Highlights field (entry_price, exit_price, pnl)
   - Calculates percentage differences

3. **Metrics Comparison**
   - Sharpe ratio
   - Max drawdown
   - Win rate
   - Total return

### Expected Differences

**Legitimate Differences**:
- Execution timing (±1 bar acceptable)
- Fill price micro-differences (<0.1%)
- Slippage model variations
- Commission calculation rounding

**Concerning Differences**:
- Trade count mismatch (signals not executed)
- Large P&L divergence (>5%)
- Missing trades
- Incorrect stop loss / take profit handling

## 🔬 Testing Checklist

Before claiming platforms agree:

- [ ] Trade count matches exactly
- [ ] Entry times within 1 bar
- [ ] Entry prices within 0.1%
- [ ] Exit prices within 0.1%
- [ ] Total P&L within 1%
- [ ] Stop loss triggers correct
- [ ] Take profit triggers correct
- [ ] Trailing stops work (if supported)
- [ ] Commission applied correctly
- [ ] Position sizing matches

## 📚 API Reference

### Signal Generator API

```python
class SignalGenerator(ABC):
    def __init__(self, name: str)
    def generate_signals(self, data: pl.DataFrame) -> list[Signal]
    def validate_data(self, data: pl.DataFrame) -> None
```

### Signal Object

```python
@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    action: 'BUY' | 'SELL' | 'CLOSE'
    quantity: float | None = None  # None = close all
    signal_id: str = ""  # Auto-generated
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_pct: float | None = None
```

### Platform Adapter API

```python
class PlatformAdapter(ABC):
    def run_backtest(
        self, signals: list, data: Any,
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0,
        **kwargs
    ) -> BacktestResult

    def supports_stop_loss() -> bool
    def supports_take_profit() -> bool
    def supports_trailing_stop() -> bool
```

### Trade Validator API

```python
class TradeValidator:
    def __init__(self, tolerance_pct: float = 0.1)
    def compare_results(
        self, results: dict[str, BacktestResult]
    ) -> ValidationReport
    def generate_html_report(
        self, report: ValidationReport, output_path: str
    ) -> None
```

## 🎯 Next Steps

1. **Install Dependencies**
   ```bash
   pip install polars pandas numpy vectorbt backtrader zipline-reloaded
   ```

2. **Run Component Test**
   ```bash
   python3 quick_test.py
   ```

3. **Run First Validation**
   ```bash
   python3 run_validation.py --strategy ma_cross --platforms ml4t.backtest
   ```

4. **Add Platforms Incrementally**
   ```bash
   python3 run_validation.py --strategy ma_cross --platforms ml4t.backtest,vectorbt_free
   python3 run_validation.py --strategy ma_cross --platforms ml4t.backtest,vectorbt_free,backtrader
   ```

5. **Test Multiple Strategies**
   ```bash
   python3 run_validation.py --strategy all --platforms ml4t.backtest,vectorbt_free
   ```

6. **Review HTML Reports**
   ```bash
   open results/*/validation_report.html
   ```

## 📝 Development Notes

### Design Decisions

1. **Signal Independence**: Signals are pure functions of data, never coupled to platforms
2. **Platform Adapters**: Handle platform-specific quirks and API differences
3. **Standardized Output**: All platforms return same BacktestResult format
4. **Tolerance-Based Validation**: Realistic tolerance levels (not exact match)
5. **Extensibility**: Easy to add new signals, platforms, validators

### Known Limitations

1. **Zipline**: May require custom bundle setup for some datasets
2. **VectorBT Pro**: Commercial license required
3. **Trailing Stops**: Not all platforms support natively
4. **Multi-Asset**: Current focus is single-symbol (expandable)

### Future Enhancements

- [ ] Pytest integration for automated testing
- [ ] Docker container for reproducibility
- [ ] More signal strategies (pairs trading, ML-based)
- [ ] Performance benchmarking
- [ ] CI/CD integration
- [ ] Multi-asset portfolio validation

---

**Framework Status**: ✅ Ready for initial testing
**Last Updated**: 2025-10-08
**Maintainer**: ml4t.backtest Team
