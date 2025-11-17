# Implementation Tasks: ML Data Architecture

**Work Unit**: 009_risk_management_exploration (expanded scope)
**Based On**: ML_DATA_ARCHITECTURE_PROPOSAL.md + External Review
**Status**: Ready for Implementation
**Timeline**: 8 weeks total (3 weeks added for critical fixes)

---

## Task Breakdown by Phase

### Phase 0: Design Refinement & Critical Fixes (1 week, 5 tasks, 40 hours)

**Purpose**: Address critical issues identified in external review before coding begins

---

#### TASK-DA-001: Fix Event Generation Performance
**Priority**: CRITICAL
**Effort**: 8 hours
**Dependencies**: None

**Problem**: Proposed implementation filters entire chunk 20,000 times (O(T×N) performance)

**Solution**: Use group_by for single-pass iteration

**Implementation**:
```python
# OLD (slow - O(T×N))
for ts in timestamps:
    asset_rows = asset_chunk.filter(pl.col("timestamp") == ts)

# NEW (fast - O(N))
for (ts, group) in asset_chunk.group_by("timestamp", maintain_order=True):
    # group is pre-filtered, O(1) access per timestamp
```

**Acceptance Criteria**:
- [ ] PolarsDataFeed uses group_by instead of filter loop
- [ ] Benchmark shows 10-50x speedup vs filter approach
- [ ] Unit test validates event order is preserved
- [ ] Document performance characteristics in docstring

**Files**: `src/ml4t/backtest/data/polars_feed.py`

**Validation**: Benchmark with 1M rows, measure time per event

---

#### TASK-DA-002: Add Signal Timing Validation
**Priority**: CRITICAL
**Effort**: 6 hours
**Dependencies**: None

**Problem**: No validation that signals are available before use (look-ahead bias risk)

**Solution**: Add explicit signal validation and timing checks

**Implementation**:
```python
class PolarsDataFeed:
    def __init__(self, ..., signal_available_at: str = "market_open"):
        """
        Args:
            signal_available_at: When signals become available:
                - "market_open": Signal valid from market open of signal date
                - "market_close": Signal valid from close of signal date
                - "next_day": Signal valid from open of next trading day
        """
        self.signal_timing = signal_available_at

    def _validate_signal_timing(self, signals_df, prices_df):
        """Validate signals don't leak future information."""
        # Check: signal timestamp <= first use timestamp
        first_use = prices_df.group_by("symbol").agg(pl.col("timestamp").min())
        signal_start = signals_df.group_by("symbol").agg(pl.col("timestamp").min())

        joined = first_use.join(signal_start, on="symbol", suffix="_signal")
        violations = joined.filter(pl.col("timestamp_signal") > pl.col("timestamp"))

        if len(violations) > 0:
            raise ValueError(f"Signal timing violation: {len(violations)} symbols")
```

**Acceptance Criteria**:
- [ ] Signal timing validation in PolarsDataFeed.__init__
- [ ] Configurable signal_available_at parameter
- [ ] Raises ValueError if signals timestamped after first use
- [ ] Unit tests cover all timing modes (open, close, next_day)
- [ ] Documentation explains when signals are assumed available

**Files**:
- `src/ml4t/backtest/data/polars_feed.py`
- `tests/unit/test_polars_feed_validation.py`

---

#### TASK-DA-003: Comprehensive Data Validation
**Priority**: CRITICAL
**Effort**: 10 hours
**Dependencies**: None

**Problem**: Only 1000-row sample validated (0.004% coverage)

**Solution**: Validate ALL rows with efficient Polars operations

**Implementation**:
```python
def _validate_data_quality(self, asset_data: pl.LazyFrame, context_data: pl.LazyFrame):
    """Comprehensive data quality checks."""

    # 1. Check for duplicates (ALL rows)
    duplicates = asset_data.group_by(['timestamp', 'symbol']).len().filter(pl.col('len') > 1).collect()
    if len(duplicates) > 0:
        raise ValueError(f"Found {len(duplicates)} duplicate (timestamp, symbol) pairs")

    # 2. Price sanity checks
    price_checks = asset_data.select([
        (pl.col('close') <= 0).any().alias('negative_price'),
        (pl.col('high') < pl.col('low')).any().alias('high_lt_low'),
        (pl.col('open') < pl.col('low')).any().alias('open_lt_low'),
        (pl.col('open') > pl.col('high')).any().alias('open_gt_high'),
        (pl.col('close') < pl.col('low')).any().alias('close_lt_low'),
        (pl.col('close') > pl.col('high')).any().alias('close_gt_high'),
        (pl.col('volume') < 0).any().alias('negative_volume'),
    ]).collect()

    failures = [k for k, v in price_checks.row(0, named=True).items() if v]
    if failures:
        raise ValueError(f"Price sanity check failures: {failures}")

    # 3. Timestamp ordering
    is_sorted = asset_data.select(
        pl.col('timestamp').is_sorted()
    ).collect().item()
    if not is_sorted:
        raise ValueError("Timestamps not sorted")

    # 4. Missing value checks
    nulls = asset_data.select([
        pl.col(c).is_null().sum().alias(c)
        for c in ['open', 'high', 'low', 'close', 'volume']
    ]).collect()

    null_counts = nulls.row(0, named=True)
    if any(count > 0 for count in null_counts.values()):
        raise ValueError(f"OHLCV contains nulls: {null_counts}")

    # 5. Context data validation (if provided)
    if context_data is not None:
        context_nulls = context_data.select(pl.all().is_null().sum()).collect()
        if context_nulls.row(0)[0] > 0:
            raise ValueError("Context data contains nulls")
```

**Acceptance Criteria**:
- [ ] All rows checked for duplicates (not sample)
- [ ] Price sanity: high >= low, open/close within range, positive prices
- [ ] Timestamp ordering validated
- [ ] Missing value detection in OHLCV
- [ ] Context data validation
- [ ] Configurable validation levels (fast/standard/paranoid)
- [ ] Clear error messages with details
- [ ] Unit tests for each validation type

**Files**:
- `src/ml4t/backtest/data/polars_feed.py`
- `tests/unit/test_data_validation.py`

---

#### TASK-DA-004: Realistic Performance Benchmarks
**Priority**: CRITICAL
**Effort**: 12 hours
**Dependencies**: TASK-DA-001 (needs fixed event generation)

**Problem**: Performance claims not validated with realistic strategies

**Solution**: Create benchmark suite with actual strategy logic

**Implementation**:
See TASK-DA-015 (Benchmark Suite) for full details.

Quick benchmarks for this task:
- Simple strategy (no orders): Measure pure event loop throughput
- Medium strategy (1% order rate): Typical ML strategy
- Complex strategy (10% order rate): Active trading

**Acceptance Criteria**:
- [ ] Benchmark simple strategy: 100-200k events/sec
- [ ] Benchmark medium strategy: 50-100k events/sec
- [ ] Benchmark complex strategy: 10-30k events/sec
- [ ] Document actual performance vs claims
- [ ] Update proposal with realistic numbers

**Files**:
- `tests/benchmarks/benchmark_event_loop.py`
- `tests/benchmarks/benchmark_strategies.py`
- `ML_DATA_ARCHITECTURE_PROPOSAL.md` (update)

---

#### TASK-DA-005: Unified API Design
**Priority**: HIGH
**Effort**: 4 hours
**Dependencies**: None

**Problem**: Dual API (Strategy vs BatchStrategy) creates maintenance burden

**Solution**: Single Strategy class with automatic mode detection

**Implementation**:
```python
class Strategy(ABC):
    """
    Base class for strategies with flexible callback modes.

    Override ONE of these methods:
    - on_market_data(event): Per-symbol callback (simple)
    - on_timestamp_batch(timestamp, asset_batch, context): Batch callback (efficient)

    Engine auto-detects which method is overridden.
    """

    def on_market_data(self, event: MarketEvent):
        """
        Process single symbol event (simple mode).

        Override this for single-asset or small universe strategies.
        Called once per symbol per timestamp.
        """
        pass  # Optional - don't override if using batch mode

    def on_timestamp_batch(self, timestamp: datetime, asset_batch: pl.DataFrame, context: dict):
        """
        Process all symbols at timestamp (batch mode).

        Override this for multi-asset strategies (50+ symbols).
        Called once per timestamp with all assets.
        """
        pass  # Optional - don't override if using simple mode

    # Helper for batch mode to iterate events
    def iter_events(self, asset_batch: pl.DataFrame, context: dict) -> Iterator[MarketEvent]:
        """Helper: Convert batch DataFrame to MarketEvent iterator."""
        for row in asset_batch.iter_rows(named=True):
            yield MarketEvent(
                timestamp=row['timestamp'],
                symbol=row['symbol'],
                data={k: row[k] for k in ['open', 'high', 'low', 'close', 'volume']},
                signals=...,
                indicators=...,
                context=context
            )


# Engine detects mode
class BacktestEngine:
    def _detect_strategy_mode(self, strategy: Strategy):
        """Auto-detect which callback to use."""
        has_simple = strategy.on_market_data.__func__ is not Strategy.on_market_data
        has_batch = strategy.on_timestamp_batch.__func__ is not Strategy.on_timestamp_batch

        if has_simple and has_batch:
            raise ValueError("Override only one: on_market_data OR on_timestamp_batch")
        elif has_batch:
            return "batch"
        else:
            return "simple"  # Default
```

**Acceptance Criteria**:
- [ ] Single Strategy base class supports both modes
- [ ] Engine auto-detects which method overridden
- [ ] iter_events() helper for batch→simple conversion
- [ ] Error if both methods overridden
- [ ] Documentation explains mode selection
- [ ] Migration guide for existing strategies

**Files**:
- `src/ml4t/backtest/strategy/base.py`
- `src/ml4t/backtest/engine.py`
- `docs/guides/strategy_modes.md`

---

### Phase 1: Core Data Infrastructure (3 weeks, 9 tasks, 120 hours)

**Purpose**: Implement PolarsDataFeed, enhanced MarketEvent, configuration system

---

#### TASK-DA-006: Enhanced MarketEvent Dataclass
**Priority**: HIGH
**Effort**: 4 hours
**Dependencies**: None

**Implementation**:
```python
@dataclass(frozen=True)
class MarketEvent:
    """Market data event with multi-source data."""
    timestamp: datetime
    symbol: str
    data: dict[str, float]  # OHLCV
    signals: dict[str, float] = field(default_factory=dict)
    indicators: dict[str, float] = field(default_factory=dict)
    context: dict[str, float] = field(default_factory=dict)

    @property
    def open(self) -> float:
        return self.data['open']

    @property
    def high(self) -> float:
        return self.data['high']

    @property
    def low(self) -> float:
        return self.data['low']

    @property
    def close(self) -> float:
        return self.data['close']

    @property
    def volume(self) -> float:
        return self.data['volume']

    def __repr__(self) -> str:
        """Compact representation."""
        sig_str = f", {len(self.signals)} signals" if self.signals else ""
        ind_str = f", {len(self.indicators)} indicators" if self.indicators else ""
        ctx_str = f", context={list(self.context.keys())}" if self.context else ""
        return f"MarketEvent({self.timestamp}, {self.symbol}, close={self.close:.2f}{sig_str}{ind_str}{ctx_str})"
```

**Acceptance Criteria**:
- [ ] MarketEvent has signals, indicators, context dicts
- [ ] Convenience properties for OHLCV
- [ ] Frozen dataclass (immutable)
- [ ] Compact __repr__ for debugging
- [ ] Unit tests for all properties
- [ ] Backward compatible (data dict works as before)

**Files**:
- `src/ml4t/backtest/core/event.py`
- `tests/unit/test_market_event.py`

---

#### TASK-DA-007: PolarsDataFeed Core Implementation
**Priority**: HIGH
**Effort**: 16 hours
**Dependencies**: TASK-DA-001, TASK-DA-002, TASK-DA-003, TASK-DA-006

**Implementation**: Full PolarsDataFeed class as specified in proposal, with all fixes applied.

**Key Features**:
- Lazy loading (scan_parquet, not read_parquet)
- Monthly chunking (configurable)
- group_by event iteration (not filter loop)
- Comprehensive validation
- Signal timing checks
- Context data lookup

**Acceptance Criteria**:
- [ ] PolarsDataFeed loads data lazily
- [ ] Chunks processed sequentially (monthly default)
- [ ] Events generated with signals, indicators, context
- [ ] All validation checks pass (duplicates, prices, timing)
- [ ] Memory efficient (<2GB for 250 symbols × 1 year)
- [ ] Unit tests: basic flow, chunking, validation
- [ ] Integration test: 10k rows end-to-end
- [ ] Docstrings complete with examples

**Files**:
- `src/ml4t/backtest/data/polars_feed.py`
- `tests/unit/test_polars_feed.py`
- `tests/integration/test_polars_feed_large.py`

---

#### TASK-DA-008: Polars Optimizations
**Priority**: MEDIUM
**Effort**: 8 hours
**Dependencies**: TASK-DA-007

**Optimizations to Add**:

**1. File Partitioning Support**
```python
# Write partitioned by month
asset_data.write_parquet("data/", partition_by="timestamp.month")

# PolarsDataFeed auto-detects partitioning
feed = PolarsDataFeed(asset_data="data/")  # Scans all partitions
```

**2. Compression Configuration**
```python
class PolarsDataFeed:
    def __init__(self, ..., compression="zstd", compression_level=3):
        # Applied when writing intermediate results
```

**3. Projection Pushdown**
```python
# Only load needed columns
needed_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
needed_columns += self.signal_columns + self.indicator_columns

chunk = self.asset_data_lazy.select(needed_columns).filter(...).collect()
```

**4. Categorical Encoding**
```python
# Encode symbol column (250 unique → 1 byte per row)
asset_data = asset_data.with_columns(pl.col("symbol").cast(pl.Categorical))
# Savings: ~168 MB for 24M rows
```

**Acceptance Criteria**:
- [ ] Partitioned file support (auto-detect)
- [ ] Configurable compression
- [ ] Projection pushdown (only load needed columns)
- [ ] Categorical encoding for symbol column
- [ ] Benchmark shows 20-30% memory reduction
- [ ] Documentation explains optimization techniques

**Files**:
- `src/ml4t/backtest/data/polars_feed.py`
- `docs/guides/performance_optimization.md`

---

#### TASK-DA-009: Configuration Schema and Loader
**Priority**: MEDIUM
**Effort**: 12 hours
**Dependencies**: None

**Implementation**:
```python
# config/schema.py
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    asset_data: str
    context_data: str | None = None
    columns: dict[str, list[str]]

class BacktestConfig(BaseModel):
    name: str
    start_date: str
    end_date: str
    initial_capital: float
    calendar: str = "NYSE"
    timezone: str = "America/New_York"
    warm_up_days: int = 0

class ExecutionConfig(BaseModel):
    slippage: dict
    commission: dict

class ReportingConfig(BaseModel):
    trades_output: str
    portfolio_output: str
    metrics_output: str

class Config(BaseModel):
    backtest: BacktestConfig
    data: DataConfig
    execution: ExecutionConfig
    reporting: ReportingConfig
    strategy: dict

# config/loader.py
def load_config(path: str) -> Config:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
```

**Acceptance Criteria**:
- [ ] Pydantic models for all config sections
- [ ] YAML loader with validation
- [ ] JSON loader support
- [ ] Default values for optional fields
- [ ] Clear validation error messages
- [ ] Unit tests for valid/invalid configs
- [ ] Example config files (3 examples)
- [ ] Documentation: config reference

**Files**:
- `src/ml4t/backtest/config/schema.py`
- `src/ml4t/backtest/config/loader.py`
- `tests/unit/test_config.py`
- `examples/configs/simple_ml_strategy.yaml`
- `examples/configs/multi_asset_strategy.yaml`
- `examples/configs/regime_switching.yaml`

---

#### TASK-DA-010: Engine Integration with PolarsDataFeed
**Priority**: HIGH
**Effort**: 12 hours
**Dependencies**: TASK-DA-007, TASK-DA-005

**Implementation**:
- Update BacktestEngine to accept PolarsDataFeed
- Implement strategy mode detection (simple vs batch)
- Route events appropriately

**Acceptance Criteria**:
- [ ] Engine accepts PolarsDataFeed as data_feed parameter
- [ ] Auto-detects strategy mode (simple/batch)
- [ ] Dispatches events correctly for each mode
- [ ] Backward compatible with existing DataFeed interface
- [ ] Integration test: full backtest with PolarsDataFeed
- [ ] Performance comparable to old DataFeed

**Files**:
- `src/ml4t/backtest/engine.py`
- `tests/integration/test_engine_polars_feed.py`

---

#### TASK-DA-011: Strategy Helper Methods
**Priority**: MEDIUM
**Effort**: 8 hours
**Dependencies**: TASK-DA-005

**Add Helper Methods**:
```python
class Strategy:
    # Position queries
    def get_position(self, symbol: str) -> Position | None:
        """Get current position for symbol."""

    def get_all_positions(self) -> pl.DataFrame:
        """Get all positions as DataFrame."""

    # Order generation helpers
    def buy(self, symbol: str, quantity: int | None = None, percent: float | None = None):
        """Generate buy order."""

    def sell(self, symbol: str, quantity: int | None = None, percent: float | None = None):
        """Generate sell order."""

    def close_position(self, symbol: str):
        """Close entire position."""

    def calculate_position_size(self, price: float, atr: float, risk_pct: float = 0.02) -> int:
        """Calculate position size based on volatility."""
```

**Acceptance Criteria**:
- [ ] Helper methods added to Strategy base class
- [ ] get_all_positions() returns Polars DataFrame
- [ ] buy/sell support both quantity and percent
- [ ] calculate_position_size() uses ATR for sizing
- [ ] Unit tests for all helpers
- [ ] Documentation with examples

**Files**:
- `src/ml4t/backtest/strategy/base.py`
- `tests/unit/test_strategy_helpers.py`

---

#### TASK-DA-012: DataPreparer Utility
**Priority**: MEDIUM
**Effort**: 16 hours
**Dependencies**: TASK-DA-007

**Purpose**: Simplify user data preparation workflow

**Implementation**:
```python
from ml4t.backtest.data import DataPreparer

prep = DataPreparer()
prep.add_prices("prices.parquet", freq="1min")
prep.add_signals("signals.parquet", freq="1day", valid_from="market_open")
prep.add_indicators(["rsi:14", "macd", "atr:14"])  # Auto-computed
prep.add_context(["VIX", "^GSPC"])

prep.validate_and_save(
    asset_output="asset_data.parquet",
    context_output="context_data.parquet"
)
```

**Acceptance Criteria**:
- [ ] DataPreparer class with fluent API
- [ ] add_prices() loads OHLCV data
- [ ] add_signals() with frequency and valid_from
- [ ] add_indicators() auto-computes from prices
- [ ] add_context() handles macro data
- [ ] Automatic join_asof for different frequencies
- [ ] Comprehensive validation before save
- [ ] Unit tests for each method
- [ ] Example notebook demonstrating workflow

**Files**:
- `src/ml4t/backtest/data/preparer.py`
- `tests/unit/test_data_preparer.py`
- `examples/notebooks/01_data_preparation.ipynb`

---

#### TASK-DA-013: Cross-Library Schema Compatibility
**Priority**: MEDIUM
**Effort**: 8 hours
**Dependencies**: TASK-DA-006

**Purpose**: Ensure ml4t.backtest outputs work with ml4t.eval

**Implementation**:
1. Define standard schemas in ml4t.core (if doesn't exist, create)
2. Update trade recording to use standard schema
3. Validate compatibility with ml4t.eval

**Acceptance Criteria**:
- [ ] Standard trade schema defined (shared across ml4t.*)
- [ ] Standard portfolio schema defined
- [ ] ml4t.backtest outputs conform to standard
- [ ] Integration test: backtest → ml4t.eval analysis
- [ ] Documentation: schema reference

**Files**:
- `../../../ml4t/core/schemas.py` (or create)
- `src/ml4t/backtest/execution/trade_tracker.py` (update)
- `tests/integration/test_eval_integration.py`

---

#### TASK-DA-014: Validation Modes (Fast/Standard/Paranoid)
**Priority**: LOW
**Effort**: 6 hours
**Dependencies**: TASK-DA-003

**Implementation**:
```python
class PolarsDataFeed:
    def __init__(self, ..., validation: str = "standard"):
        """
        Args:
            validation: 'fast' | 'standard' | 'paranoid'
        """
        self.validation_level = validation

    def _validate_data_quality(self):
        if self.validation_level == "fast":
            # Schema + sample checks (1s)
            self._validate_schema()
            self._validate_sample(1000)
        elif self.validation_level == "standard":
            # Full duplicate check + price sanity (10s)
            self._validate_schema()
            self._validate_all_duplicates()
            self._validate_price_sanity()
        elif self.validation_level == "paranoid":
            # Everything + signal timing + staleness (60s)
            self._validate_schema()
            self._validate_all_duplicates()
            self._validate_price_sanity()
            self._validate_signal_timing()
            self._validate_signal_staleness()
```

**Acceptance Criteria**:
- [ ] Three validation levels implemented
- [ ] Fast: <2 seconds
- [ ] Standard: <15 seconds
- [ ] Paranoid: <90 seconds
- [ ] Documentation explains trade-offs
- [ ] Unit tests for each level

**Files**:
- `src/ml4t/backtest/data/polars_feed.py`
- `tests/unit/test_validation_levels.py`

---

### Phase 2: Trade Recording Enhancement (1 week, 3 tasks, 40 hours)

**Purpose**: Comprehensive trade records with signals, context, cost breakdown

---

#### TASK-DA-015: Enhanced Trade Schema
**Priority**: HIGH
**Effort**: 12 hours
**Dependencies**: TASK-DA-006

**Problem**: Use struct types instead of JSON for context

**Implementation**:
```python
trades_df = pl.DataFrame({
    # Trade identification
    "trade_id": pl.Utf8,
    "symbol": pl.Utf8,

    # Entry details
    "entry_timestamp": pl.Datetime,
    "entry_price": pl.Float64,
    "entry_quantity": pl.Float64,
    "entry_commission": pl.Float64,
    "entry_slippage": pl.Float64,

    # Entry signals/indicators
    "entry_signal_value": pl.Float64,
    "entry_signal_confidence": pl.Float64,
    "entry_rsi": pl.Float64,  # Example indicator
    "entry_atr": pl.Float64,  # Example indicator

    # Entry context (separate columns, not JSON)
    "entry_vix": pl.Float64,
    "entry_spy": pl.Float64,

    # Exit details
    "exit_timestamp": pl.Datetime,
    "exit_price": pl.Float64,
    "exit_quantity": pl.Float64,
    "exit_commission": pl.Float64,
    "exit_slippage": pl.Float64,
    "exit_reason": pl.Utf8,  # Categorical

    # Exit signals/context
    "exit_signal_value": pl.Float64,
    "exit_vix": pl.Float64,
    "exit_spy": pl.Float64,

    # P&L metrics
    "pnl_gross": pl.Float64,
    "pnl_net": pl.Float64,
    "pnl_percent": pl.Float64,
    "total_cost": pl.Float64,
    "holding_period_bars": pl.Int64,
    "holding_period_days": pl.Float64,

    # Risk metrics
    "stop_loss_level": pl.Float64,
    "take_profit_level": pl.Float64,
    "max_favorable_excursion": pl.Float64,
    "max_adverse_excursion": pl.Float64,
    "max_favorable_pct": pl.Float64,
    "max_adverse_pct": pl.Float64,
})
```

**Acceptance Criteria**:
- [ ] Trade schema includes all fields from proposal
- [ ] Context stored as separate columns (not JSON)
- [ ] Signals and indicators recorded at entry/exit
- [ ] exit_reason is categorical
- [ ] Schema validated with Pydantic
- [ ] Documentation: field descriptions

**Files**:
- `src/ml4t/backtest/execution/trade_tracker.py`
- `src/ml4t/backtest/core/schemas.py`
- `tests/unit/test_trade_schema.py`

---

#### TASK-DA-016: TradeTracker Enhancement
**Priority**: HIGH
**Effort**: 16 hours
**Dependencies**: TASK-DA-015, TASK-DA-006

**Purpose**: Update TradeTracker to record enhanced data

**Implementation**:
- Capture MarketEvent data at entry (signals, indicators, context)
- Capture MarketEvent data at exit
- Track MFE/MAE during position lifetime
- Record commission and slippage separately
- Export to Parquet with enhanced schema

**Acceptance Criteria**:
- [ ] TradeTracker captures signals at entry/exit
- [ ] Context values recorded at entry/exit
- [ ] MFE/MAE tracked during position lifetime
- [ ] Commission and slippage separated
- [ ] Parquet export with comprehensive schema
- [ ] Integration test: full backtest → trade records
- [ ] Verify all fields populated correctly

**Files**:
- `src/ml4t/backtest/execution/trade_tracker.py`
- `tests/integration/test_trade_recording.py`

---

#### TASK-DA-017: Portfolio States Time Series
**Priority**: MEDIUM
**Effort**: 12 hours
**Dependencies**: None

**Implementation**:
```python
portfolio_states_df = pl.DataFrame({
    "timestamp": pl.Datetime,
    "equity": pl.Float64,
    "cash": pl.Float64,
    "positions_value": pl.Float64,
    "num_positions": pl.Int64,
    "gross_leverage": pl.Float64,
    "net_leverage": pl.Float64,
    "daily_return": pl.Float64,
    "cumulative_return": pl.Float64,

    # Optional: Context at this timestamp
    "vix": pl.Float64,
    "spy": pl.Float64,
})
```

**Acceptance Criteria**:
- [ ] Portfolio snapshots recorded at configurable frequency
- [ ] All metrics calculated correctly
- [ ] Context data included (optional)
- [ ] Exported to Parquet
- [ ] Integration test: backtest → portfolio states
- [ ] Compatible with ml4t.eval

**Files**:
- `src/ml4t/backtest/portfolio/state.py`
- `src/ml4t/backtest/reporting/parquet.py`
- `tests/integration/test_portfolio_recording.py`

---

### Phase 3: Documentation & Examples (2 weeks, 7 tasks, 80 hours)

**Purpose**: Complete user documentation, guides, and examples

---

#### TASK-DA-018: User Guide - ML Signal Integration
**Priority**: HIGH
**Effort**: 12 hours
**Dependencies**: Phase 1 complete

**Content**:
1. Overview of ML workflow (train → predict → backtest)
2. Signal format requirements
3. Timing considerations (when are signals available?)
4. Data preparation with DataPreparer
5. Strategy implementation
6. Results analysis

**Acceptance Criteria**:
- [ ] Complete guide (3000+ words)
- [ ] Code examples for each step
- [ ] Timing diagrams
- [ ] Common pitfalls section
- [ ] Published in docs/

**Files**:
- `docs/guides/ml_signal_integration.md`

---

#### TASK-DA-019: User Guide - Multi-Asset Strategies
**Priority**: HIGH
**Effort**: 12 hours
**Dependencies**: Phase 1 complete

**Content**:
1. Multi-asset data organization
2. Universe definition
3. Simple vs batch strategy modes
4. Top-N selection strategies
5. Performance considerations
6. Memory management

**Acceptance Criteria**:
- [ ] Complete guide (3000+ words)
- [ ] Code examples
- [ ] Performance tips
- [ ] Published in docs/

**Files**:
- `docs/guides/multi_asset_strategies.md`

---

#### TASK-DA-020: Example Notebook - Single-Asset ML Strategy
**Priority**: HIGH
**Effort**: 8 hours
**Dependencies**: Phase 1 complete

**Content**:
- Download AAPL data
- Generate ML signals (simple logistic regression)
- Prepare data with DataPreparer
- Implement strategy
- Run backtest
- Analyze results

**Acceptance Criteria**:
- [ ] Executable notebook (all cells run without errors)
- [ ] Clear explanations
- [ ] Produces visualizations
- [ ] Uses real data

**Files**:
- `examples/notebooks/02_single_asset_ml_strategy.ipynb`

---

#### TASK-DA-021: Example Notebook - Multi-Asset Top 25 Strategy
**Priority**: HIGH
**Effort**: 12 hours
**Dependencies**: Phase 1 complete

**Content**:
- 250-stock universe (S&P 500 subset)
- Daily ML signals (top 25 by confidence)
- Minute execution data
- VIX filtering
- Position sizing with ATR
- Risk management (stops)
- Complete backtest
- Performance analysis

**Acceptance Criteria**:
- [ ] Executable notebook
- [ ] Realistic strategy
- [ ] Demonstrates batch mode
- [ ] Shows signal integration
- [ ] Context filtering (VIX)
- [ ] Performance metrics

**Files**:
- `examples/notebooks/03_multi_asset_top25_strategy.ipynb`

---

#### TASK-DA-022: Example Notebook - Regime Switching Strategy
**Priority**: MEDIUM
**Effort**: 8 hours
**Dependencies**: Phase 1 complete

**Content**:
- Bull/bear regime detection (VIX, moving averages)
- Different strategies per regime
- Context-dependent logic
- Regime transition handling

**Acceptance Criteria**:
- [ ] Executable notebook
- [ ] Demonstrates context usage
- [ ] Clear regime switching logic

**Files**:
- `examples/notebooks/04_regime_switching_strategy.ipynb`

---

#### TASK-DA-023: API Documentation
**Priority**: MEDIUM
**Effort**: 16 hours
**Dependencies**: Phase 1 complete

**Content**:
- Auto-generated API docs (Sphinx)
- PolarsDataFeed reference
- MarketEvent reference
- Strategy reference
- Configuration reference
- All public APIs documented

**Acceptance Criteria**:
- [ ] Sphinx documentation builds without errors
- [ ] All public APIs have docstrings
- [ ] Code examples in docstrings
- [ ] Published to docs/api/

**Files**:
- `docs/api/` (multiple files)
- `docs/conf.py` (Sphinx config)

---

#### TASK-DA-024: Migration Guide
**Priority**: MEDIUM
**Effort**: 8 hours
**Dependencies**: Phase 1 complete

**Content**:
- How to migrate from old DataFeed to PolarsDataFeed
- Strategy changes (helper methods)
- Configuration migration
- What's deprecated
- Breaking changes

**Acceptance Criteria**:
- [ ] Step-by-step migration instructions
- [ ] Code examples (before/after)
- [ ] Deprecation timeline
- [ ] Published in docs/

**Files**:
- `docs/guides/migration_to_polars_feed.md`

---

#### TASK-DA-025: Performance Optimization Guide
**Priority**: LOW
**Effort** 4 hours
**Dependencies**: TASK-DA-008

**Content**:
- Polars optimization techniques
- File partitioning
- Compression tuning
- Memory profiling
- Performance benchmarking

**Acceptance Criteria**:
- [ ] Guide with examples
- [ ] Benchmarking instructions
- [ ] Published in docs/

**Files**:
- `docs/guides/performance_optimization.md`

---

### Phase 4: Benchmark Suite (Parallel, 1 week, 3 tasks, 40 hours)

**Purpose**: Validate performance claims and create regression tests

---

#### TASK-DA-026: Event Loop Benchmarks
**Priority**: HIGH
**Effort**: 12 hours
**Dependencies**: TASK-DA-007

**Benchmarks**:
1. Pure event generation (no strategy logic)
2. Event generation with signals/indicators/context
3. Chunked vs full loading
4. group_by vs filter performance

**Acceptance Criteria**:
- [ ] Benchmark suite using pytest-benchmark
- [ ] Tests with 10k, 100k, 1M, 10M rows
- [ ] Memory profiling (memory_profiler)
- [ ] Results documented
- [ ] Regression tests (performance doesn't degrade)

**Files**:
- `tests/benchmarks/benchmark_event_loop.py`
- `tests/benchmarks/RESULTS.md`

---

#### TASK-DA-027: Strategy Benchmarks
**Priority**: HIGH
**Effort**: 16 hours
**Dependencies**: TASK-DA-010

**Benchmarks**:
1. Simple strategy (no orders): Measure pure overhead
2. Passive strategy (buy & hold): Minimal logic
3. Medium strategy (1% order rate): Typical ML strategy
4. Active strategy (10% order rate): High-frequency logic

**Test Configuration**:
- 250 symbols, 1 year, minute data (24.5M events)
- Run on typical hardware (document specs)
- Measure: events/sec, memory peak, total time

**Acceptance Criteria**:
- [ ] 4 strategy benchmarks implemented
- [ ] Simple: >100k events/sec
- [ ] Medium: 50-100k events/sec
- [ ] Active: 10-30k events/sec
- [ ] Memory: <2 GB peak
- [ ] Results documented with hardware specs
- [ ] Comparison vs VectorBT, Backtrader (if possible)

**Files**:
- `tests/benchmarks/benchmark_strategies.py`
- `tests/benchmarks/STRATEGY_RESULTS.md`

---

#### TASK-DA-028: Scalability Benchmarks
**Priority**: MEDIUM
**Effort**: 12 hours
**Dependencies**: TASK-DA-010

**Benchmarks**:
- Scale symbols: 10, 50, 100, 250, 500, 1000
- Scale time: 1 month, 3 months, 6 months, 1 year, 2 years
- Scale frequency: daily, hourly, 1min
- Memory vs scale
- Time vs scale

**Acceptance Criteria**:
- [ ] Benchmark results for various scales
- [ ] Identify bottlenecks (symbols vs time vs frequency)
- [ ] Document where performance degrades
- [ ] Memory profiling
- [ ] Results with scaling graphs

**Files**:
- `tests/benchmarks/benchmark_scalability.py`
- `tests/benchmarks/SCALABILITY_RESULTS.md`

---

## Summary

### Total Effort Breakdown

| Phase | Tasks | Hours | Weeks |
|-------|-------|-------|-------|
| **Phase 0: Critical Fixes** | 5 | 40 | 1 |
| **Phase 1: Core Infrastructure** | 9 | 120 | 3 |
| **Phase 2: Trade Recording** | 3 | 40 | 1 |
| **Phase 3: Documentation** | 7 | 80 | 2 |
| **Phase 4: Benchmarks** | 3 | 40 | 1 (parallel) |
| **Total** | **27 tasks** | **320 hours** | **8 weeks** |

### Critical Path

```
Phase 0 → Phase 1 → Phase 2 ─┬─→ Phase 3
            └──────────────────→ Phase 4
```

Phase 4 (benchmarks) can run in parallel with Phase 2-3 once Phase 1 is complete.

### Priority Tasks (Must Complete)

**P0 (Blockers):**
- TASK-DA-001: Fix event generation
- TASK-DA-002: Signal timing validation
- TASK-DA-003: Comprehensive validation
- TASK-DA-004: Realistic benchmarks

**P1 (Core Features):**
- TASK-DA-005: Unified API
- TASK-DA-006: Enhanced MarketEvent
- TASK-DA-007: PolarsDataFeed core
- TASK-DA-010: Engine integration
- TASK-DA-015: Enhanced trade schema
- TASK-DA-016: TradeTracker enhancement

### Success Criteria

**By end of Phase 0:**
- [ ] Event generation 10-50x faster than naive implementation
- [ ] No look-ahead bias possible (validation prevents it)
- [ ] Data quality guaranteed (all rows validated)
- [ ] Realistic performance targets documented

**By end of Phase 1:**
- [ ] PolarsDataFeed functional with real data
- [ ] Full backtest completes successfully
- [ ] Memory <2 GB for 250 symbols × 1 year
- [ ] Configuration-driven backtests work

**By end of Phase 2:**
- [ ] Trade records include all signals/context
- [ ] Portfolio states exported
- [ ] Compatible with ml4t.eval

**By end of Phase 3:**
- [ ] 3 example notebooks executable
- [ ] User guides complete
- [ ] API documentation published

**By end of Phase 4:**
- [ ] Performance validated: 10-30k events/sec for realistic strategies
- [ ] Scalability limits documented
- [ ] Regression tests in place

---

**Status**: Ready for Phase 0 Implementation
**Next Action**: Begin TASK-DA-001 (Fix Event Generation Performance)
