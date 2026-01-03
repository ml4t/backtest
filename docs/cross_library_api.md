# ML4T Cross-Library API Specification

**Version**: 1.0.0
**Status**: Production
**Last Updated**: 2026-01-02

## Overview

This document specifies the data interchange formats between ml4t libraries, designed for:
- **Consistency**: Identical schemas across Python, Numba, and Rust implementations
- **Efficiency**: Parquet-native types for zero-copy interop
- **Diagnostics Integration**: Seamless handoff to ml4t-diagnostic (like zipline→pyfolio)

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ML4T DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ml4t.data          ml4t.engineer        ml4t.backtest                  │
│  ──────────────────────────────────────────────────────────────         │
│  ┌─────────┐        ┌─────────────┐      ┌────────────────┐             │
│  │  OHLCV  │───────▶│  Features   │─────▶│ BacktestResult │             │
│  │  Data   │        │  + Signals  │      │                │             │
│  └─────────┘        └─────────────┘      └───────┬────────┘             │
│                                                  │                      │
│                                                  ▼                      │
│                                          ┌──────────────┐               │
│                                          │   Parquet    │               │
│                                          │   Storage    │               │
│                                          └──────┬───────┘               │
│                                                 │                       │
│                           ┌─────────────────────┼─────────────────────┐ │
│                           │                     │                     │ │
│                           ▼                     ▼                     ▼ │
│                    ┌────────────┐      ┌──────────────┐    ┌──────────┐ │
│                    │ Portfolio  │      │    Trade     │    │  Trade   │ │
│                    │  Analysis  │      │   Analysis   │    │   SHAP   │ │
│                    └────────────┘      └──────────────┘    └──────────┘ │
│                           │                     │                │      │
│                           └─────────────────────┴────────────────┘      │
│                                          │                              │
│                                          ▼                              │
│                                   ml4t.diagnostic                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Schemas

### 1. Trades DataFrame Schema

The canonical schema for completed trades. All implementations must produce this exact schema.

| Column | Type | Description | Nullable |
|--------|------|-------------|----------|
| `asset` | String | Asset identifier (e.g., "AAPL", "BTC-USD") | No |
| `entry_time` | Datetime[μs] | Position entry timestamp | No |
| `exit_time` | Datetime[μs] | Position exit timestamp | No |
| `entry_price` | Float64 | Average entry price | No |
| `exit_price` | Float64 | Average exit price | No |
| `quantity` | Float64 | Position size (negative for short) | No |
| `direction` | String | "long" or "short" | No |
| `pnl` | Float64 | Realized P&L (after costs) | No |
| `pnl_percent` | Float64 | P&L as percentage of entry value | No |
| `bars_held` | Int32 | Number of bars position was held | No |
| `commission` | Float64 | Total commission (entry + exit) | No |
| `slippage` | Float64 | Total slippage (entry + exit) | No |
| `mfe` | Float64 | Max favorable excursion (best unrealized %) | No |
| `mae` | Float64 | Max adverse excursion (worst unrealized %) | No |
| `exit_reason` | String | Why position was closed (see ExitReason) | No |

**Parquet Settings**:
- Compression: `zstd` (default)
- Row group size: 100,000 rows

### 2. Equity DataFrame Schema

Time series of portfolio value and derived metrics.

| Column | Type | Description | Nullable |
|--------|------|-------------|----------|
| `timestamp` | Datetime[μs] | Bar timestamp | No |
| `equity` | Float64 | Portfolio value | No |
| `return` | Float64 | Period return (0 for first bar) | No |
| `cumulative_return` | Float64 | Return from start | No |
| `drawdown` | Float64 | Current drawdown (negative) | No |
| `high_water_mark` | Float64 | Running maximum equity | No |

### 3. Daily P&L DataFrame Schema

Day-level aggregation for calendar or session-aligned analysis.

| Column | Type | Description | Nullable |
|--------|------|-------------|----------|
| `date` | Date | Trading date | No |
| `open_equity` | Float64 | First bar equity | No |
| `close_equity` | Float64 | Last bar equity | No |
| `high_equity` | Float64 | Maximum equity | No |
| `low_equity` | Float64 | Minimum equity | No |
| `pnl` | Float64 | Daily P&L | No |
| `return_pct` | Float64 | Daily return percentage | No |
| `cumulative_return` | Float64 | Cumulative return from start | No |
| `num_bars` | Int32 | Bars in this day | No |

## Exit Reasons

Standardized enum for cross-library compatibility:

| Value | Description |
|-------|-------------|
| `signal` | Strategy signal triggered exit |
| `stop_loss` | Stop-loss order triggered |
| `take_profit` | Take-profit order triggered |
| `trailing_stop` | Trailing stop triggered |
| `time_stop` | Time-based exit (max bars held) |
| `end_of_data` | Position closed at backtest end |

## Diagnostic Integration

### PortfolioAnalysis (pyfolio replacement)

```python
from ml4t.backtest import Engine
from ml4t.diagnostic.evaluation import PortfolioAnalysis

# Run backtest
result = engine.run()

# Direct integration
analysis = PortfolioAnalysis(
    returns=result.to_returns_series().to_numpy(),
    dates=result.to_equity_dataframe()["timestamp"],
)
metrics = analysis.compute_summary_stats()
tear_sheet = analysis.create_tear_sheet()
```

### TradeAnalysis (trade-level diagnostics)

```python
from ml4t.backtest import Engine
from ml4t.diagnostic.integration import TradeRecord
from ml4t.diagnostic.evaluation import TradeAnalysis

# Run backtest
result = engine.run()

# Convert to TradeRecord format
trade_records = [TradeRecord(**r) for r in result.to_trade_records()]

# Analyze
analyzer = TradeAnalysis(trade_records)
worst = analyzer.worst_trades(n=20)
stats = analyzer.compute_statistics()
```

### TradeShapAnalyzer (ML error diagnostics)

```python
from ml4t.backtest import Engine, enrich_trades_with_signals
from ml4t.diagnostic.evaluation import TradeShapAnalyzer

# Run backtest with ML signals
result = engine.run()
trades_df = result.to_trades_dataframe()

# Enrich with signal values at entry/exit times
enriched = enrich_trades_with_signals(
    trades_df,
    signals_df,  # Your ML features DataFrame
    signal_columns=["momentum", "rsi", "ml_score"]
)

# SHAP analysis on worst trades
# (TradeShapAnalyzer expects feature values at trade time)
```

## Signal Enrichment

Post-process join for adding ML features to trades without storing during execution.

```python
from ml4t.backtest import enrich_trades_with_signals

# trades_df: from result.to_trades_dataframe()
# signals_df: Your ML features with timestamp column

enriched = enrich_trades_with_signals(
    trades_df,
    signals_df,
    signal_columns=["momentum", "rsi", "ml_score"],
    timestamp_col="timestamp",
    asset_col="asset",  # For multi-asset
)

# Result has: entry_momentum, exit_momentum, entry_rsi, exit_rsi, etc.
```

## Storage Format

### Directory Structure

```
{backtest_results}/
├── trades.parquet        # Trades DataFrame
├── equity.parquet        # Equity DataFrame
├── daily_pnl.parquet     # Daily P&L DataFrame
├── metrics.json          # Summary metrics
└── config.yaml           # Backtest configuration (optional)
```

### Usage

```python
# Save
result.to_parquet("./results/my_backtest")

# Load
loaded = BacktestResult.from_parquet("./results/my_backtest")
```

## Implementation Checklist

For Numba/Rust implementations to be compatible:

- [ ] Produce exact Trades DataFrame schema
- [ ] Produce exact Equity DataFrame schema
- [ ] Use microsecond datetime precision
- [ ] Use IEEE 754 Float64 for all numeric values
- [ ] Use UTF-8 String for text fields
- [ ] Support `zstd` Parquet compression
- [ ] Implement `enrich_trades_with_signals()` equivalent
- [ ] Output `exit_reason` from standardized enum

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-02 | Initial specification |
