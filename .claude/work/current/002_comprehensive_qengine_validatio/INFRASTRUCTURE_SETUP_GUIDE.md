# Infrastructure Setup Guide - Phase 0

## Overview
This guide documents the installation and configuration of isolated virtual environments for cross-framework validation of QEngine against VectorBT Pro, Zipline-Reloaded, and Backtrader.

## Virtual Environment Strategy
Each framework is installed in an isolated virtual environment to prevent dependency conflicts:

```
.venv-vectorbt     # VectorBT Pro (Python 3.12.3)
.venv-zipline      # Zipline-Reloaded (TBD)
.venv-backtrader   # Backtrader (TBD)
.venv              # QEngine (existing, Polars-based)
```

**Rationale**: Clean dependency management, easier debugging, framework-specific Python versions if needed.

---

## TASK-001: VectorBT Pro Installation

### Status: ⚠️ BLOCKED - Incomplete Source Distribution

### Steps Completed

#### 1. Virtual Environment Creation
```bash
python3 -m venv .venv-vectorbt
```
- ✅ Created at: `/home/stefan/ml4t/backtest/.venv-vectorbt/`
- ✅ Python version: 3.12.3

#### 2. Upgrade Build Tools
```bash
.venv-vectorbt/bin/pip install --upgrade pip setuptools wheel
```
- ✅ pip: 25.2
- ✅ setuptools: 80.9.0
- ✅ wheel: 0.45.1

#### 3. Install VectorBT Pro Base
```bash
.venv-vectorbt/bin/pip install -e resources/vectorbt.pro-main/
```
**Dependencies installed:**
- numpy 2.3.3
- pandas 2.3.3
- numba 0.62.1
- scipy 1.16.2
- scikit-learn 1.7.2
- schedule, requests, tqdm, dateparser, imageio
- typing_extensions, mypy_extensions, attrs
- websocket-client

#### 4. Install Data Dependencies
```bash
.venv-vectorbt/bin/pip install -e "resources/vectorbt.pro-main/[data-base]"
```
**Additional dependencies:**
- tables 3.10.2 (HDF5 support)
- SQLAlchemy 2.0.43
- duckdb 1.4.0
- pyarrow 21.0.0
- yfinance 0.2.66
- python-binance 1.0.29
- beautifulsoup4, aiohttp, websockets

#### 5. Source Code Fixes Applied
Fixed missing pandas type imports in `resources/vectorbt.pro-main/vectorbtpro/_typing.py`:

```python
# Added at line 37-38:
from pandas import (
    DatetimeIndex,
    Index,
    IndexSlice,     # ← Added
    MultiIndex,     # ← Added
    PeriodIndex,
    Series,
    Timestamp,
)
```

### Critical Issue: Missing `data` Module

**Problem:**
- VectorBT Pro source at `resources/vectorbt.pro-main/` is incomplete
- Missing `vectorbtpro/data/` directory entirely
- Import chain fails at `vectorbtpro/ohlcv/accessors.py:102`:
  ```python
  from vectorbtpro.data.base import OHLCDataMixin  # ← Module does not exist
  ```

**Cannot currently:**
- ❌ Import vectorbtpro at all (`import vectorbtpro` fails)
- ❌ Use high-level VectorBT Pro APIs
- ❌ Run backtests through VectorBT Pro framework

**May still work:**
- ⚠️ Direct imports of low-level numba functions (untested)
- ⚠️ Specific module imports that don't depend on `data` (untested)

### Resolution Required

#### Recommended Actions (Priority Order):

**1. Obtain Complete VectorBT Pro Distribution (BEST)**
- Check if VectorBT Pro is available on PyPI: `pip install vectorbtpro`
- Use `quantgpt.chat` to ask about proper installation
- Contact user for complete source distribution
- Verify vectorbtpro.data module exists before proceeding

**2. Alternative: Use Open-Source VectorBT**
```bash
.venv-vectorbt/bin/pip install vectorbt
```
- Open-source version (not Pro)
- May lack some Pro features but sufficient for validation
- Well-documented and actively maintained

**3. Create Temporary Stub (NOT RECOMMENDED)**
- Create dummy `vectorbtpro/data/` module to satisfy imports
- High risk of silent failures and incorrect validation results

### Testing Plan (Once Resolved)

**Hello-World Test:**
```python
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# Simple MA crossover backtest
price = pd.Series([100, 102, 101, 103, 105, 104, 106, 108],
                  index=pd.date_range('2020-01-01', periods=8))

fast_ma = vbt.MA.run(price, window=2)
slow_ma = vbt.MA.run(price, window=4)

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

portfolio = vbt.Portfolio.from_signals(price, entries, exits)

print(f"Final value: ${portfolio.final_value():.2f}")
print(f"Total return: {portfolio.total_return():.2%}")
print(f"Trades: {portfolio.trades.count()}")
```

**Success Criteria:**
- [ ] Import vectorbtpro without errors
- [ ] Create and run simple MA crossover strategy
- [ ] Generate portfolio statistics
- [ ] Verify results are reasonable (non-zero returns, valid trade count)

---

## TASK-002: Zipline-Reloaded Installation

### Status: ⏳ PENDING (Waiting for TASK-001 resolution)

### Planned Steps

#### 1. Research Zipline-Reloaded Requirements
- Check Python version compatibility
- Review data bundle requirements
- Identify timezone handling needs

#### 2. Create Virtual Environment
```bash
python3 -m venv .venv-zipline
.venv-zipline/bin/pip install --upgrade pip setuptools wheel
```

#### 3. Install Zipline-Reloaded
```bash
.venv-zipline/bin/pip install zipline-reloaded
```

#### 4. Configure Data Bundle
```bash
.venv-zipline/bin/zipline ingest -b quandl
# OR: Custom bundle for daily equities from ~/ml4t/projects/
```

#### 5. Hello-World Test
```python
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol
import pandas as pd

def initialize(context):
    context.asset = symbol('AAPL')

def handle_data(context, data):
    order_target_percent(context.asset, 1.0)

perf = run_algorithm(
    start=pd.Timestamp('2020-01-01', tz='UTC'),
    end=pd.Timestamp('2020-12-31', tz='UTC'),
    initialize=initialize,
    handle_data=handle_data,
    capital_base=10000
)

print(f"Final portfolio value: {perf['portfolio_value'].iloc[-1]}")
```

**Known Issues to Address:**
- Timezone complexity (always use UTC)
- Data bundle format (custom loader needed for Polars data)
- Slow performance (126x slower than QEngine)

---

## TASK-003: Backtrader Installation

### Status: ⏳ PENDING

### Planned Steps

#### 1. Create Virtual Environment
```bash
python3 -m venv .venv-backtrader
.venv-backtrader/bin/pip install --upgrade pip setuptools wheel
```

#### 2. Install Backtrader
```bash
.venv-backtrader/bin/pip install backtrader
.venv-backtrader/bin/pip install backtrader[plotting]  # Optional: matplotlib for plots
```

#### 3. Hello-World Test
```python
import backtrader as bt
import pandas as pd

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.buy()
        else:
            if self.data.close[0] < self.sma[0]:
                self.close()

cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy)

# Load data (custom feed needed for Polars data)
data = bt.feeds.PandasData(dataname=price_df)
cerebro.adddata(data)

cerebro.broker.setcash(10000.0)
cerebro.run()

print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
```

**Known Issues to Address:**
- Signal execution bugs (document as limitations)
- 6x slower than QEngine
- Feed format conversion needed

---

## TASK-007: Universal Data Loader

### Status: ⏳ PENDING

### Design Plan

#### Data Sources to Support
1. **Daily US Equities**: `~/ml4t/projects/daily_us_equities/equity_prices_enhanced_1962_2025.parquet`
2. **NASDAQ100 Minute Bars**: `~/ml4t/projects/nasdaq100_minute_bars/{2021,2022}.parquet`
3. **Crypto Futures/Spot**: `~/ml4t/projects/crypto_futures/data/{futures,spot}/{BTC,ETH}.parquet`
4. **SPY Order Flow**: `~/ml4t/projects/spy_order_flow/spy_features.parquet`
5. **Tick Data**: `~/ml4t/projects/tick_data/`

#### Loader Architecture
```python
class UniversalDataLoader:
    """Load data from all sources and convert to framework-specific formats"""

    @staticmethod
    def load_daily_equities(symbols: list[str], start: str, end: str) -> pl.DataFrame:
        """Load daily equity data"""
        path = "~/ml4t/projects/daily_us_equities/equity_prices_enhanced_1962_2025.parquet"
        df = pl.read_parquet(path)
        return df.filter(
            (pl.col('symbol').is_in(symbols)) &
            (pl.col('date') >= start) &
            (pl.col('date') <= end)
        )

    @staticmethod
    def to_vectorbt_format(df: pl.DataFrame) -> pd.DataFrame:
        """Convert Polars to VectorBT format (pandas with specific columns)"""
        return df.to_pandas().set_index('date')

    @staticmethod
    def to_zipline_format(df: pl.DataFrame) -> pd.DataFrame:
        """Convert to Zipline bundle format"""
        pdf = df.to_pandas()
        pdf.index = pd.to_datetime(pdf['date']).tz_localize('UTC')
        return pdf[['open', 'high', 'low', 'close', 'volume']]

    @staticmethod
    def to_backtrader_format(df: pl.DataFrame) -> pd.DataFrame:
        """Convert to Backtrader feed format"""
        return df.to_pandas().rename(columns={
            'date': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })

    @staticmethod
    def to_qengine_format(df: pl.DataFrame) -> pl.DataFrame:
        """QEngine uses Polars directly"""
        return df
```

---

## Troubleshooting

### VectorBT Pro Import Errors
**Issue**: `ModuleNotFoundError: No module named 'vectorbtpro.data'`
- **Cause**: Incomplete source distribution
- **Solution**: Obtain complete VectorBT Pro package or use open-source vectorbt

### Zipline Timezone Errors
**Issue**: `ValueError: Timezone mismatch`
- **Cause**: Zipline requires UTC timezone
- **Solution**: Always use `tz='UTC'` for timestamps

### Backtrader Signal Execution Bugs
**Issue**: Missing trades compared to other frameworks
- **Cause**: Known Backtrader bugs
- **Solution**: Document as limitation, don't use for correctness baseline

### Dependency Conflicts
**Issue**: Package version conflicts between frameworks
- **Solution**: Use isolated virtual environments (already implemented)

---

## Next Steps

1. **IMMEDIATE**: Resolve VectorBT Pro installation issue
   - Try `pip install vectorbtpro` from PyPI
   - Use quantgpt.chat for guidance
   - Consider open-source vectorbt alternative

2. **AFTER TASK-001**: Proceed with Zipline-Reloaded (TASK-002)

3. **AFTER TASK-002**: Proceed with Backtrader (TASK-003)

4. **PARALLEL WITH INSTALLATIONS**: Develop Universal Data Loader (TASK-007)

5. **AFTER ALL COMPLETE**: Baseline verification test (TASK-008)

---

## Progress Tracking

- [x] TASK-001 environment created
- [x] TASK-001 dependencies installed
- [ ] TASK-001 ✗ **BLOCKED** - incomplete source
- [ ] TASK-002 (pending)
- [ ] TASK-003 (pending)
- [ ] TASK-007 (pending)
- [ ] TASK-008 (pending)
- [ ] TASK-009 documentation
