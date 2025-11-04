# Exploration Summary: Cross-Platform Backtest Validation

## Executive Summary

Comprehensive validation framework is feasible with structured phased approach. QEngine is production-ready (September 2025) with 159 passing tests and 100% agreement with VectorBT on basic scenarios. Rich data available in `../projects/`. Main risks: vectorbt-pro license availability and platform execution model differences.

## Codebase Analysis

### QEngine Current State (Production-Ready)

**Core Components** (from `src/qengine/`):
- `engine.py`: BacktestEngine, BacktestResults - Main entry point
- `core/`: Event bus, Clock (time control), base abstractions
- `data/`: Data feed abstractions (ParquetDataFeed confirmed)
- `execution/`: Broker, order types (Market, Limit, Stop, Bracket)
- `portfolio/`: Position tracking, P&L calculations
- `strategy/`: Strategy base class with lifecycle hooks
- `reporting/`: Performance metrics and reporting

**Validated Features**:
- ✅ Event-driven architecture with no data leakage
- ✅ Execution delay for temporal accuracy
- ✅ Multi-feed synchronization (critical for validation)
- ✅ P&L calculations verified for all asset classes
- ✅ Cash constraints and corporate actions handled
- ✅ 159 unit tests, 8,552 trades/sec performance
- ✅ **100% agreement with VectorBT** (excellent baseline)

**Integration Points**:
- Data loading: `ParquetDataFeed` - Can load from ../projects/*.parquet
- Strategy interface: `Strategy.on_market_event()` - Where signals translate to orders
- Results: `BacktestResults` - Standardized output format

### Available Data (../projects/)

**High-Quality Datasets**:
1. **Daily US Equities**: `equity_prices_enhanced_1962_2025.parquet`
   - 63 years of data, excellent for long-term strategies
   - Recommended for initial validation

2. **NASDAQ-100 Minute Bars**: `2021.parquet`, `2022.parquet`
   - Intraday data for higher frequency testing
   - 2 years of minute-level data

3. **Crypto Data**: BTC/ETH futures and spot
   - Alternative asset class validation
   - Tests 24/7 market handling

4. **SPY Order Flow**: Microstructure data
   - Advanced validation scenarios
   - Phase 2+ consideration

**Recommendation**: Start with `daily_us_equities` for reliability and platform compatibility.

### Platform Comparison Matrix

| Platform | Type | Fill Model | Data Format | Complexity | Status |
|----------|------|------------|-------------|------------|--------|
| **QEngine** | Event-driven | Realistic broker sim | Polars/Parquet | Medium | ✅ Production-ready |
| **Zipline** | Event-driven | Bar-based fills | Pandas DataFrame | High | ⚠️ Complex deps |
| **VectorBT** | Vectorized | Configurable fills | Pandas/NumPy | Low | ✅ Free version OK |
| **Backtrader** | Event-driven | Flexible broker | Custom feeds | Medium | ✅ Stable |

**Key Insight**: VectorBT should be primary reference (already 100% validated with QEngine).

## Implementation Architecture

### Proposed Directory Structure

```
tests/validation/
├── README.md                          # Setup and usage instructions
├── requirements.txt                   # Pinned dependencies
│
├── data/                              # Data loading and standardization
│   ├── loaders.py                     # Load from ../projects/
│   ├── standardizers.py               # Convert to platform formats
│   └── cache/                         # Cached transformed data
│
├── signals/                           # Platform-independent signals
│   ├── base.py                        # SignalGenerator ABC
│   ├── ma_crossover.py                # Simple MA strategy
│   ├── mean_reversion.py              # Bollinger/RSI strategy
│   ├── multi_factor.py                # Complex signals
│   └── test_signals.py                # Signal unit tests
│
├── adapters/                          # Platform-specific adapters
│   ├── base.py                        # Adapter interface
│   ├── qengine_adapter.py             # QEngine integration
│   ├── zipline_adapter.py             # Zipline integration
│   ├── vectorbt_adapter.py            # VectorBT integration
│   ├── backtrader_adapter.py          # Backtrader integration
│   └── test_adapters.py               # Adapter tests
│
├── validators/                        # Result comparison
│   ├── signal_validator.py            # Verify signal consistency
│   ├── trade_validator.py             # Compare trade execution
│   ├── pnl_validator.py               # Compare P&L
│   ├── metrics_validator.py           # Performance metrics
│   └── report_generator.py            # HTML/Markdown reports
│
├── test_cases/                        # Validation scenarios
│   ├── test_01_buy_hold.py            # Baseline
│   ├── test_02_ma_crossover.py        # Simple signals
│   ├── test_03_mean_reversion.py      # Frequent trading
│   ├── test_04_multi_symbol.py        # Portfolio
│   ├── test_05_long_short.py          # Short positions
│   └── ...                            # Additional scenarios
│
├── results/                           # Test outputs (gitignored)
│   └── YYYY-MM-DD_HH-MM-SS/          # Timestamped runs
│       ├── qengine_results.json
│       ├── zipline_results.json
│       ├── vectorbt_results.json
│       ├── backtrader_results.json
│       ├── validation_summary.json
│       └── report.html                # Visual comparison
│
└── run_validation.py                  # Main orchestrator
```

### Component Design

#### 1. Signal Generator (Platform-Independent)

```python
# signals/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Signal:
    """Platform-agnostic trading signal"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'CLOSE'
    quantity: float | None  # None = close all
    signal_id: str

class SignalGenerator(ABC):
    """Base class for signal generation"""

    @abstractmethod
    def generate_signals(self, data: pl.DataFrame) -> list[Signal]:
        """Generate signals from OHLCV data"""
        pass
```

#### 2. Platform Adapter

```python
# adapters/base.py
class PlatformAdapter(ABC):
    """Interface for platform-specific backtesting"""

    @abstractmethod
    def run_backtest(
        self,
        signals: list[Signal],
        data: Any,  # Platform-specific format
        config: dict
    ) -> BacktestResult:
        """Execute backtest with given signals"""
        pass

@dataclass
class BacktestResult:
    """Standardized results across platforms"""
    platform: str
    trades: list[Trade]
    pnl_series: pl.DataFrame
    metrics: dict
    execution_time: float
```

#### 3. Validator

```python
# validators/trade_validator.py
class TradeValidator:
    """Compare trades across platforms"""

    def compare_trades(
        self,
        results: dict[str, BacktestResult]
    ) -> ValidationReport:
        """Compare trade execution across platforms"""
        # 1. Check trade count
        # 2. Compare entry/exit times
        # 3. Compare fill prices
        # 4. Identify discrepancies
        pass
```

## Phased Implementation Plan

### Phase 1: Foundation (Week 1) - RECOMMENDED START
**Goal**: Validate architecture with minimal scope

**Tasks**:
1. Set up directory structure
2. Implement signal generator interface + MA crossover
3. Load data from `../projects/daily_us_equities/`
4. Implement QEngine adapter (easiest, already understood)
5. Implement VectorBT adapter (reference platform, 100% validated)
6. Create buy-and-hold test case
7. Build basic validator (trade count, P&L)

**Deliverable**: QEngine vs VectorBT working on 1 signal, 1 dataset

**Risk Mitigation**:
- Start with platforms we know work
- Simplest signal type
- Proven data source
- Validates full architecture end-to-end

### Phase 2: Expand Platforms (Week 2)
**Goal**: Add remaining platforms

**Tasks**:
1. Install and test zipline-reloaded
2. Implement zipline adapter
3. Install and test backtrader
4. Implement backtrader adapter
5. Run buy-and-hold test across all 4 platforms
6. Document platform-specific differences

**Deliverable**: All 4 platforms validated on baseline test

**Blockers**:
- Zipline installation issues (complex deps)
- VectorBT-Pro license (fallback to free version)

### Phase 3: Expand Test Cases (Week 3)
**Goal**: Comprehensive validation scenarios

**Tasks**:
1. Implement mean reversion signal generator
2. Implement multi-factor signal generator
3. Create 5-7 additional test cases
4. Run full validation suite
5. Analyze and document differences

**Deliverable**: 7-10 test cases passing validation

### Phase 4: Production Readiness (Week 4)
**Goal**: Robust, reusable framework

**Tasks**:
1. Enhance report generation (HTML with charts)
2. Add statistical analysis of differences
3. Create Docker container for reproducibility
4. Write comprehensive documentation
5. Add CI/CD integration (if desired)

**Deliverable**: Production-ready validation framework

## Key Technical Decisions

### 1. Execution Semantics Standardization

To ensure fair comparison, configure ALL platforms with:
- **Fill Price**: Next bar open (most common, least ambiguous)
- **Commission**: 0.1% per trade (or 0 for pure signal validation)
- **Slippage**: 0 bps (or fixed 5 bps if desired)
- **Position Sizing**: Fixed quantity or percentage
- **Order Type**: Market orders only (simplest)

**Rationale**: Eliminate execution model differences, isolate signal logic.

### 2. Data Format Strategy

Each platform expects different data formats:
- QEngine: Polars DataFrame
- Zipline: Pandas with DatetimeIndex, specific columns
- VectorBT: Pandas or NumPy arrays
- Backtrader: Feed objects or Pandas

**Solution**: Data loader outputs all formats once, cache for reuse.

```python
class DataStandardizer:
    def load_and_convert(self, parquet_path: str) -> dict:
        """Load once, convert to all formats"""
        df_polars = pl.read_parquet(parquet_path)
        return {
            'polars': df_polars,
            'pandas': df_polars.to_pandas(),
            'zipline': self._to_zipline_format(df_polars),
            'backtrader': self._to_backtrader_format(df_polars)
        }
```

### 3. Signal Independence Enforcement

Signals MUST be generated WITHOUT any backtester:
- Input: Raw OHLCV data (Polars DataFrame)
- Output: List of Signal objects
- Process: Pure function, no side effects
- Validation: Same data → same signals every time

```python
# Example enforcement
def test_signal_determinism():
    """Verify signals are deterministic"""
    data = load_test_data()
    signals1 = generator.generate_signals(data)
    signals2 = generator.generate_signals(data)
    assert signals1 == signals2  # Must be identical
```

### 4. Comparison Tolerance Levels

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| Signal count | 0% | Must match exactly |
| Trade count | 0% | Same signals = same trades |
| Trade times | ±1 bar | Platform timing differences |
| Fill prices | ±0.1% | Rounding, tie-breaking |
| Total P&L | ±1% | Accumulation of small diffs |

Differences beyond tolerance → Investigation required.

## Risk Assessment

### HIGH Priority Risks

1. **VectorBT-Pro License**
   - Impact: Cannot validate against commercial features
   - Mitigation: Use free VectorBT (sufficient for validation)
   - Status: Acceptable fallback

2. **Execution Model Divergence**
   - Impact: Platforms may have legitimate differences
   - Mitigation: Start with simplest execution rules, document diffs
   - Status: Expected, manageable

3. **QEngine Missing Features**
   - Impact: May discover unimplemented features during testing
   - Mitigation: QEngine is production-ready, most features done
   - Status: Low probability given recent fixes

### MEDIUM Priority Risks

4. **Zipline Installation Complexity**
   - Impact: Difficult to install on some systems
   - Mitigation: Docker container, document workarounds
   - Status: Manageable with containerization

5. **Data Compatibility**
   - Impact: Some data may not work with all platforms
   - Mitigation: Start with daily equities (most compatible)
   - Status: Low risk with careful data selection

6. **Scope Creep**
   - Impact: Too many test cases, takes too long
   - Mitigation: Phased approach, start with 2-3 cases
   - Status: Controlled with discipline

## Next Steps

### Immediate Actions (This Session)

1. ✅ **Data verification**: Confirm `daily_us_equities` is accessible and usable
2. ✅ **Platform installation check**: Verify backtrader, vectorbt can be installed
3. ⏭️ **Create detailed plan**: Break Phase 1 into specific tasks for `/plan`

### Recommended Workflow

```bash
# After exploration complete:
/plan --from-requirements  # Create detailed implementation plan

# Then execute:
/next  # Start with first task (likely: set up directory structure)
/next  # Implement signal interface
/next  # Load data
# ... continue until Phase 1 complete
```

### Questions for Clarification

1. **VectorBT License**: Do you have vectorbt-pro license? (Free version is fine)
2. **Priority**: Which platforms are most important? (Can prioritize if needed)
3. **Timeline**: Is 4-week phased approach acceptable? (Can accelerate if needed)
4. **Scope**: Should we start with Phase 1 only, then reassess?

## Success Indicators

- ✅ Requirements captured and documented
- ✅ QEngine codebase understood (production-ready)
- ✅ Data availability confirmed (rich datasets in ../projects/)
- ✅ Architecture designed (modular, extensible)
- ✅ Risks identified and mitigated
- ✅ Phased approach defined (incremental delivery)
- ✅ Next steps clear (ready for `/plan`)

## Conclusion

This is a **well-scoped, achievable validation project** with:
- Clear objective (cross-platform validation)
- Strong foundation (QEngine production-ready, 100% VectorBT agreement)
- Good data (multiple datasets available)
- Manageable risks (mitigated with phased approach)

**Recommendation**: Proceed with `/plan` to create detailed Phase 1 tasks, then execute with `/next`.
