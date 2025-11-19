# Framework Adapter Implementation Guide

**Purpose**: Guide for implementing multi-asset Top-N momentum adapters for all 4 frameworks

**Context**: Extends existing single-asset validation (`test_cross_framework_alignment.py`) to multi-asset portfolios

## Signal Format

### Input: Pre-Computed Signals
```python
# DataFrame from compute_momentum_signals()
timestamp           symbol      signal
2020-01-22          STOCK00     1      # BUY
2020-01-22          STOCK01     1      # BUY
2020-01-22          STOCK02     1      # BUY
2020-01-22          STOCK03     1      # BUY
2020-01-22          STOCK04     1      # BUY
2020-02-11          STOCK00     -1     # SELL (dropped out of top 5)
2020-02-11          STOCK15     1      # BUY (entered top 5)
...
```

### Target Configuration
- **Initial Capital**: $100,000
- **Position Size**: 20% per stock (5 stocks max, equal weight)
- **Commission**: 0.1% per trade
- **Slippage**: 0.1% per trade
- **Fill Timing**: Next-open (realistic) or same-close (for matching)

## 1. ml4t.backtest Adapter

### File
`tests/validation/frameworks/qengine_adapter.py`

### Current Status
Single-asset only (`run_with_signals()` expects single DataFrame)

### Implementation Steps

#### Step 1: Multi-Asset Data Feed
```python
def run_with_signals(
    self,
    data: dict[str, pd.DataFrame],  # Changed: dict of symbol -> OHLCV
    signals: pd.DataFrame,          # Pre-computed signals
    config: FrameworkConfig,
) -> ValidationResult:
    """Run backtest with multi-asset data and pre-computed signals."""

    # Create Polars DataFrames for each symbol
    polars_data = {}
    for symbol, df in data.items():
        polars_data[symbol] = pl.from_pandas(df.reset_index())

    # Create PolarsDataFeed for multi-asset
    feeds = {
        symbol: PolarsDataFeed(
            data=pl_df,
            asset_id=symbol,
            signals=signals[signals['symbol'] == symbol]  # Filter signals per asset
        )
        for symbol, pl_df in polars_data.items()
    }
```

#### Step 2: Signal-Based Strategy
```python
class TopNMomentumStrategy(Strategy):
    """Executes pre-computed rotation signals."""

    def __init__(self, signals_df: pd.DataFrame, target_weight: float = 0.20):
        super().__init__(name="TopN_Momentum")
        self.signals = signals_df
        self.target_weight = target_weight
        self.signal_idx = 0

    def on_market_data(self, event: MarketEvent):
        """Check for signals at this timestamp."""
        current_time = event.timestamp

        # Get signals for this timestamp
        day_signals = self.signals[
            self.signals['timestamp'] == current_time
        ]

        for _, signal_row in day_signals.iterrows():
            symbol = signal_row['symbol']
            signal = signal_row['signal']

            if signal == 1:  # BUY
                # Calculate equal weight position
                target_value = self.broker.portfolio.equity * self.target_weight
                current_price = event.close  # Assuming we get close price
                shares = target_value / current_price

                self.buy(asset_id=symbol, quantity=shares)

            elif signal == -1:  # SELL
                # Close position
                position = self.broker.get_position(symbol)
                if position and position.quantity > 0:
                    self.sell(asset_id=symbol, quantity=position.quantity)
```

#### Step 3: Trade Extraction
```python
def _extract_trades(self, broker: SimulationBroker) -> list[TradeRecord]:
    """Extract trades from broker's trade tracker."""
    trades = []

    for fill_event in broker.fill_events:
        trades.append(TradeRecord(
            timestamp=fill_event.timestamp,
            action='BUY' if fill_event.quantity > 0 else 'SELL',
            quantity=abs(fill_event.quantity),
            price=fill_event.price,
            value=abs(fill_event.quantity * fill_event.price),
            commission=fill_event.commission,
        ))

    return trades
```

### Source Code References
- `src/ml4t/backtest/data/polars_feed.py:183-222` - Multi-feed setup
- `src/ml4t/backtest/strategy/base.py:384-422` - Strategy interface
- `src/ml4t/backtest/execution/broker.py:472-601` - Position tracking

### Estimated Time
3-4 hours

---

## 2. VectorBT Pro Adapter

### File
`tests/validation/frameworks/vectorbt_adapter.py`

### Current Status
Single-asset `Portfolio.from_signals()` only

### Implementation Steps

#### Step 1: Multi-Asset Price Matrix
```python
def run_with_signals(
    self,
    data: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    config: FrameworkConfig,
) -> ValidationResult:
    """Run VectorBT with multi-asset signals."""

    # Create price matrix (columns = symbols, index = timestamps)
    prices = pd.DataFrame({
        symbol: df['close']
        for symbol, df in data.items()
    })

    # Create entry/exit matrices (boolean)
    entries = pd.DataFrame(
        False,
        index=prices.index,
        columns=prices.columns
    )
    exits = pd.DataFrame(
        False,
        index=prices.index,
        columns=prices.columns
    )

    # Fill from signals
    for _, row in signals.iterrows():
        timestamp = row['timestamp']
        symbol = row['symbol']
        signal = row['signal']

        if signal == 1:
            entries.loc[timestamp, symbol] = True
        elif signal == -1:
            exits.loc[timestamp, symbol] = True
```

#### Step 2: Portfolio Creation
```python
    # Run VectorBT backtest
    import vectorbtpro as vbt

    portfolio = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        size=config.initial_capital * 0.20 / prices,  # Equal weight per signal
        size_type='targetvalue',  # Target portfolio percentage
        fees=config.commission_pct,
        slippage=config.slippage_pct,
        cash_sharing=True,  # Share cash across all assets
        group_by=True,  # Group all assets into one portfolio
        init_cash=config.initial_capital,
    )
```

#### Step 3: Trade Extraction
```python
    # Extract trades from VectorBT
    trades_df = portfolio.trades.records_readable

    trades = [
        TradeRecord(
            timestamp=row['Entry Timestamp'],
            action='BUY' if row['Size'] > 0 else 'SELL',
            quantity=abs(row['Size']),
            price=row['Entry Price'],
            value=abs(row['Size'] * row['Entry Price']),
            commission=row['Fees'],
        )
        for _, row in trades_df.iterrows()
    ]
```

### Source Code References
- `resources/vectorbt.pro-main/vectorbtpro/portfolio/base.py:1834-2103` - `from_signals()` method
- `resources/vectorbt.pro-main/vectorbtpro/portfolio/nb/from_signals.py:45-312` - Numba execution
- `resources/vectorbt.pro-main/vectorbtpro/portfolio/trades.py:127-456` - Trade extraction

### Estimated Time
2-3 hours

---

## 3. Backtrader Adapter

### File
`tests/validation/frameworks/backtrader_adapter.py`

### Current Status
Single-asset only

### Implementation Steps

#### Step 1: Multi-Data Feed Setup
```python
def run_with_signals(
    self,
    data: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    config: FrameworkConfig,
) -> ValidationResult:
    """Run Backtrader with multi-asset signals."""

    import backtrader as bt

    cerebro = bt.Cerebro()

    # Add data feeds for each symbol
    for symbol, df in data.items():
        bt_data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
        )
        cerebro.adddata(bt_data, name=symbol)

    # Add signal-based strategy
    cerebro.addstrategy(
        SignalBasedStrategy,
        signals_df=signals,
        target_weight=0.20,
    )

    # Configure broker
    cerebro.broker.setcash(config.initial_capital)
    cerebro.broker.setcommission(commission=config.commission_pct)
    cerebro.broker.set_coc(config.backtrader_coc)  # Cheat-On-Close for matching
```

#### Step 2: Signal-Based Strategy
```python
class SignalBasedStrategy(bt.Strategy):
    """Executes pre-computed signals in Backtrader."""

    params = (
        ('signals_df', None),
        ('target_weight', 0.20),
    )

    def __init__(self):
        self.signals = self.params.signals_df
        self.order_refs = {}  # Track pending orders

    def next(self):
        """Called for each bar."""
        current_date = self.datas[0].datetime.date(0)

        # Get signals for this date
        day_signals = self.signals[
            self.signals['timestamp'].dt.date == current_date
        ]

        for _, signal_row in day_signals.iterrows():
            symbol = signal_row['symbol']
            signal = signal_row['signal']

            # Find data feed for this symbol
            data_feed = None
            for data in self.datas:
                if data._name == symbol:
                    data_feed = data
                    break

            if data_feed is None:
                continue

            if signal == 1:  # BUY
                # Calculate target value (20% of portfolio)
                target_value = self.broker.getvalue() * self.params.target_weight

                # Place order
                self.order_target_value(
                    data=data_feed,
                    target=target_value,
                )

            elif signal == -1:  # SELL
                # Close position
                self.close(data=data_feed)
```

#### Step 3: Trade Extraction
```python
    # Run backtest
    results = cerebro.run()

    # Extract trades from Cerebro's analyzer
    # Backtrader stores trades in strategy.trades
    trades = []
    for trade_record in results[0].trades:
        trades.append(TradeRecord(
            timestamp=trade_record.dtopen,
            action='BUY' if trade_record.size > 0 else 'SELL',
            quantity=abs(trade_record.size),
            price=trade_record.price,
            value=abs(trade_record.value),
            commission=trade_record.commission,
        ))
```

### Source Code References
- `resources/backtrader-master/backtrader/cerebro.py:145-892` - Multi-data setup
- `resources/backtrader-master/backtrader/brokers/bbroker.py:893-903` - COC execution
- `resources/backtrader-master/backtrader/trade.py:38-156` - Trade tracking

### Estimated Time
4-5 hours

---

## 4. Zipline Adapter

### File
`tests/validation/frameworks/zipline_adapter.py`

### Current Status
Skeleton only, bundle issue documented

### Implementation Steps

#### Step 1: Create Custom Bundle
**File**: `tests/validation/bundles/momentum_test_bundle.py`

```python
"""Custom Zipline bundle for multi-asset momentum test."""

import pandas as pd
import numpy as np
from pathlib import Path

BUNDLE_DATA_DIR = Path(__file__).parent
BUNDLE_DATA_FILE = BUNDLE_DATA_DIR / 'momentum_test.h5'

def prepare_bundle_data(data: dict[str, pd.DataFrame]):
    """
    Prepare test data for Zipline bundle ingestion.

    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
    """
    # Create HDF5 file with all stocks
    with pd.HDFStore(BUNDLE_DATA_FILE, mode='w') as store:
        # Write equities metadata
        equities = pd.DataFrame([
            {
                'sid': i,
                'symbol': symbol,
                'asset_name': symbol,
                'exchange': 'NYSE',
            }
            for i, symbol in enumerate(sorted(data.keys()))
        ])
        store.put('equities', equities)

        # Write OHLCV data for each stock
        for sid, symbol in enumerate(sorted(data.keys())):
            df = data[symbol].copy()

            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')

            store.put(str(sid), df)

        # No splits/dividends for test data
        store.put('splits', pd.DataFrame())
        store.put('dividends', pd.DataFrame())

    print(f"✅ Created bundle data: {BUNDLE_DATA_FILE}")
    print(f"   {len(equities)} stocks, {len(df)} days")

def momentum_test_ingest(environ, asset_db_writer, minute_bar_writer,
                         daily_bar_writer, adjustment_writer, calendar,
                         start_session, end_session, cache, show_progress,
                         output_dir):
    """
    Zipline bundle ingest function.

    Called by: zipline ingest -b momentum_test
    """
    # Follow pattern from tests/validation/bundles/validation_ingest.py
    # (See validation_ingest.py lines 85-186)
    pass  # Implementation follows validation_ingest.py pattern exactly

# Register bundle
from zipline.data.bundles import register
register(
    'momentum_test',
    momentum_test_ingest,
    calendar_name='NYSE',
)
```

#### Step 2: Register Bundle
**File**: Add to `~/.zipline/extension.py` or set `ZIPLINE_EXTENSION_PATH`

```python
import sys
from pathlib import Path

# Add test bundle to path
test_bundle_path = Path('/home/stefan/ml4t/software/backtest/tests/validation/bundles')
sys.path.insert(0, str(test_bundle_path))

# Import will register bundle
from momentum_test_bundle import momentum_test_ingest
```

#### Step 3: Ingest Bundle
```bash
# Prepare data first
python -c "
from test_integrated_framework_alignment import generate_multi_asset_data
from momentum_test_bundle import prepare_bundle_data

data = generate_multi_asset_data()
prepare_bundle_data(data)
"

# Ingest into Zipline
zipline ingest -b momentum_test
```

#### Step 4: Signal-Based Algorithm
```python
def run_with_signals(
    self,
    data: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    config: FrameworkConfig,
) -> ValidationResult:
    """Run Zipline with multi-asset signals."""

    # Prepare bundle (call prepare_bundle_data)
    from momentum_test_bundle import prepare_bundle_data
    prepare_bundle_data(data)

    # Ingest bundle
    import subprocess
    subprocess.run(['zipline', 'ingest', '-b', 'momentum_test'], check=True)

    # Define algorithm
    def initialize(context):
        """Initialize algorithm."""
        context.signals = signals.copy()
        context.signal_idx = 0
        context.symbols = {
            symbol: symbol(symbol)  # Zipline symbol lookup
            for symbol in signals['symbol'].unique()
        }

        # Set costs
        from zipline.finance.commission import PerShare
        from zipline.finance.slippage import FixedSlippage

        set_commission(PerShare(cost=config.commission_pct * 100))
        set_slippage(FixedSlippage(spread=config.slippage_pct))

    def handle_data(context, data):
        """Process each bar."""
        current_date = data.current_dt.date()

        # Get signals for this date
        day_signals = context.signals[
            context.signals['timestamp'].dt.date == current_date
        ]

        for _, signal_row in day_signals.iterrows():
            symbol = signal_row['symbol']
            signal = signal_row['signal']

            asset = context.symbols.get(symbol)
            if asset is None:
                continue

            if signal == 1:  # BUY
                # Target 20% of portfolio
                target_value = context.portfolio.portfolio_value * 0.20
                order_target_value(asset, target_value)

            elif signal == -1:  # SELL
                order_target_value(asset, 0)  # Close position

    # Run algorithm
    from zipline import run_algorithm

    results = run_algorithm(
        start=signals['timestamp'].min(),
        end=signals['timestamp'].max(),
        initialize=initialize,
        handle_data=handle_data,
        capital_base=config.initial_capital,
        data_frequency='daily',
        bundle='momentum_test',
    )

    # Extract trades
    trades = self._extract_trades(results)

    return ValidationResult(
        framework='Zipline',
        strategy='TopN_Momentum',
        initial_capital=config.initial_capital,
        final_value=results['portfolio_value'].iloc[-1],
        total_return=(results['portfolio_value'].iloc[-1] / config.initial_capital - 1) * 100,
        num_trades=len(trades),
        trades=trades,
    )
```

### Source Code References
- `resources/zipline-reloaded-main/zipline/data/bundles/core.py:45-128` - Bundle registration
- `resources/zipline-reloaded-main/zipline/finance/execution.py:67-213` - Order execution
- `tests/validation/bundles/validation_ingest.py:85-186` - Bundle ingest pattern

### Estimated Time
5-6 hours (bundle creation + algorithm + testing)

---

## Testing Strategy

### Phase 1: Individual Framework Testing
Test each adapter in isolation before comparison:

```python
# Test ml4t.backtest only
pytest tests/validation/test_integrated_framework_alignment.py::TestIntegratedFrameworkAlignment::test_qengine_execution -v -s --tb=short

# Verify:
# - Trades executed
# - Portfolio value calculated
# - No errors
```

### Phase 2: Pairwise Comparison
Compare frameworks incrementally:

```python
# 2-way: ml4t.backtest vs VectorBT
# Expect: <0.5% variance

# 3-way: + Backtrader
# Expect: <0.5% variance across all 3

# 4-way: + Zipline
# Expect: <0.5% variance across all 4
```

### Phase 3: Full 4-Way Validation
```python
pytest tests/validation/test_integrated_framework_alignment.py::TestIntegratedFrameworkAlignment::test_all_frameworks_alignment -v -s
```

Expected output:
```
================================================================================
4-WAY FRAMEWORK VALIDATION: Top-N Momentum (25 stocks, 1 year)
================================================================================
Signals: 98 total
Config: 0.1% commission, 0.1% slippage
================================================================================

Running qengine...
   ✅ Completed in 2.341s
      Final Value: $105,234.56
      Trades: 98

Running VectorBT...
   ✅ Completed in 0.523s
      Final Value: $105,189.23
      Trades: 98

Running Backtrader...
   ✅ Completed in 3.876s
      Final Value: $105,301.45
      Trades: 96

Running Zipline...
   ✅ Completed in 4.123s
      Final Value: $105,276.89
      Trades: 98

================================================================================
COMPARISON RESULTS
================================================================================
Framework       Final Value        Return    Trades  Time (s)
--------------------------------------------------------------------------------
qengine         $  105,234.56      5.23%        98     2.341
VectorBT        $  105,189.23      5.19%        98     0.523
Backtrader      $  105,301.45      5.30%        96     3.876
Zipline         $  105,276.89      5.28%        98     4.123

Variance Statistics:
  Value Range: $112.22 (0.1122%)
  Return Range: 0.1122%
  Trade Count Range: 2
================================================================================

✅ 4-WAY VALIDATION PASSED - All frameworks produce equivalent results!
   Variance: 0.1122% (<0.5% threshold)
   Trade alignment: ±2 trades
```

## Common Issues & Solutions

### Issue 1: Quantity Mismatches
**Problem**: Fractional shares vs whole shares
**Solution**: Set `fractional_shares=True` in `FrameworkConfig`

### Issue 2: End-of-Backtest Divergence
**Problem**: Some frameworks auto-close final positions, others don't
**Solution**: Explicitly close all positions in signals (already done in `compute_momentum_signals()`)

### Issue 3: Fill Price Differences
**Problem**: Open vs close vs VWAP fills
**Solution**: Use `FrameworkConfig.for_matching()` preset for same-bar close fills across all

### Issue 4: Commission/Slippage Calculation Order
**Problem**: Some frameworks apply commission before slippage, others after
**Solution**: Use small values (0.1% each) to minimize order effects

## Source Code Investigation Protocol

**MANDATORY**: When frameworks produce different results:

1. **DO NOT GUESS** - Read the actual source code
2. **Cite line numbers** - Example: "VectorBT fills at close (base.py:2045)"
3. **Compare implementations** - Show code snippets from both frameworks
4. **Document in test output** - Add findings to assertion messages

**Example**:
```python
assert value_variance < 0.5, f"""
Value variance {value_variance:.4f}% exceeds 0.5% threshold

INVESTIGATION (cite source code):
- ml4t.backtest: Fills at next open (fill_simulator.py:329)
- VectorBT: Fills at signal bar close (from_signals.py:167)
- Difference: 1-bar fill timing delay causes price gap

FIX: Set config.fill_timing='same_close' for both frameworks
"""
```

## Validation Checklist

Before marking test as COMPLETE:

- [ ] All 4 adapters implemented
- [ ] Each adapter tested in isolation (trades executed, no errors)
- [ ] Zipline bundle created and ingests successfully
- [ ] All 4 frameworks run without errors
- [ ] Final value variance <0.5%
- [ ] Trade count within ±5 trades (document if outside)
- [ ] Commission/slippage applied equivalently (verify with small trades)
- [ ] Any systematic differences documented with source code citations
- [ ] Test output includes comparison table
- [ ] Test passes consistently (run 3x to verify reproducibility)

---

**Total Estimated Time**: 20-26 hours

**Priority Order**:
1. ml4t.backtest (3-4h) - Baseline
2. VectorBT (2-3h) - Fast comparison
3. Backtrader (4-5h) - Production validation
4. Zipline (5-6h + 2-3h bundle) - Complete validation

**Success Metric**: All 4 frameworks within 0.5% variance with documented explanations for any systematic differences.
