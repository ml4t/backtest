# Trade Recording Schema Documentation

## Overview

The `MLTradeRecord` schema provides comprehensive trade recording with ML signals, risk management fields, and market context for post-backtest analysis.

## Schema Fields

### Core Trade Details
- **trade_id** (int): Unique trade identifier
- **asset_id** (str): Asset symbol/identifier
- **direction** (str): Trade direction ("long" or "short")

### Entry Details
- **entry_dt** (datetime): Entry timestamp
- **entry_price** (float): Entry fill price
- **entry_quantity** (float): Entry quantity (always positive)
- **entry_commission** (float): Entry commission cost
- **entry_slippage** (float): Entry slippage cost
- **entry_order_id** (str): Entry order identifier

### Exit Details
- **exit_dt** (datetime | None): Exit timestamp
- **exit_price** (float | None): Exit fill price
- **exit_quantity** (float | None): Exit quantity
- **exit_commission** (float): Exit commission cost
- **exit_slippage** (float): Exit slippage cost
- **exit_order_id** (str): Exit order identifier
- **exit_reason** (ExitReason): Reason for exit (enum)

### Trade Metrics
- **pnl** (float | None): Net profit/loss (after all costs)
- **return_pct** (float | None): Return percentage on capital at risk
- **duration_bars** (int | None): Number of bars held
- **duration_seconds** (float | None): Hold time in seconds

### ML Signals (Entry)
- **ml_score_entry** (float | None): ML model score/prediction at entry
- **predicted_return_entry** (float | None): Predicted return at entry
- **confidence_entry** (float | None): Model confidence at entry (0-1)

### ML Signals (Exit)
- **ml_score_exit** (float | None): ML model score/prediction at exit
- **predicted_return_exit** (float | None): Predicted return at exit
- **confidence_exit** (float | None): Model confidence at exit (0-1)

### Technical Indicators (Entry)
- **atr_entry** (float | None): Average True Range at entry
- **volatility_entry** (float | None): Realized volatility at entry
- **momentum_entry** (float | None): Momentum indicator at entry
- **rsi_entry** (float | None): RSI indicator at entry

### Technical Indicators (Exit)
- **atr_exit** (float | None): Average True Range at exit
- **volatility_exit** (float | None): Realized volatility at exit
- **momentum_exit** (float | None): Momentum indicator at exit
- **rsi_exit** (float | None): RSI indicator at exit

### Risk Management
- **stop_loss_price** (float | None): Stop-loss price (if set)
- **take_profit_price** (float | None): Take-profit price (if set)
- **risk_reward_ratio** (float | None): Risk/reward ratio at entry
- **position_size_pct** (float | None): Position size as % of portfolio

### Market Context (Entry)
- **vix_entry** (float | None): VIX level at entry
- **market_regime_entry** (str | None): Market regime at entry
- **sector_performance_entry** (float | None): Sector performance at entry

### Market Context (Exit)
- **vix_exit** (float | None): VIX level at exit
- **market_regime_exit** (str | None): Market regime at exit
- **sector_performance_exit** (float | None): Sector performance at exit

### Additional Data
- **metadata** (dict): Dictionary for any additional custom fields

## Exit Reasons

The `ExitReason` enum defines standard exit reasons:

- **SIGNAL**: Normal signal-based exit
- **STOP_LOSS**: Stop-loss triggered
- **TAKE_PROFIT**: Take-profit triggered
- **TIME_STOP**: Maximum hold time exceeded
- **RISK_RULE**: Risk rule triggered (VIX, volatility, etc)
- **POSITION_SIZE**: Position sizing constraint
- **END_OF_DATA**: Backtest ended with open position
- **MANUAL**: Manual exit (e.g., user intervention)
- **UNKNOWN**: Unknown or unspecified

## Usage Examples

### Creating a Trade Record

```python
from datetime import datetime, timezone
from ml4t.backtest.reporting import MLTradeRecord, ExitReason

trade = MLTradeRecord(
    trade_id=1,
    asset_id="BTC",
    direction="long",
    entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
    entry_price=50000.0,
    entry_quantity=1.0,
    entry_commission=10.0,
    exit_dt=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    exit_price=51000.0,
    exit_quantity=1.0,
    exit_commission=10.0,
    exit_reason=ExitReason.TAKE_PROFIT,
    pnl=970.0,
    return_pct=1.94,
    duration_bars=2,
    # ML signals
    ml_score_entry=0.85,
    confidence_entry=0.9,
    # Technical indicators
    atr_entry=1000.0,
    rsi_entry=65.0,
    # Risk management
    stop_loss_price=49000.0,
    take_profit_price=52000.0,
    # Market context
    vix_entry=15.0,
    market_regime_entry="bull",
)
```

### Converting to Polars DataFrame

```python
from ml4t.backtest.reporting import trades_to_polars

# Convert list of trades to DataFrame
df = trades_to_polars([trade1, trade2, trade3])

# Analyze trades
winning_trades = df.filter(pl.col("pnl") > 0)
avg_return = df["return_pct"].mean()
```

### Exporting to Parquet

```python
from ml4t.backtest.reporting import export_parquet

# Export with compression
export_parquet(
    trades=[trade1, trade2, trade3],
    path="trades.parquet",
    compression="zstd",
    compression_level=3
)
```

### Incremental Writes

```python
from ml4t.backtest.reporting import append_trades

# Append new trades to existing file
append_trades(
    new_trades=[trade4, trade5],
    path="trades.parquet"
)
```

### Reading Back from Parquet

```python
from ml4t.backtest.reporting import import_parquet, polars_to_trades

# Read as DataFrame
df = import_parquet("trades.parquet")

# Convert to trade records
trades = polars_to_trades(df)
```

## Analysis Examples

### ML Model Evaluation

```python
import polars as pl
from ml4t.backtest.reporting import import_parquet

# Load trades
df = import_parquet("trades.parquet")

# Analyze ML predictions vs actual returns
df = df.with_columns([
    (pl.col("pnl") / (pl.col("entry_price") * pl.col("entry_quantity"))).alias("actual_return")
])

# Compare predicted vs actual
comparison = df.select([
    "trade_id",
    "predicted_return_entry",
    "actual_return",
    "confidence_entry"
])
```

### Risk Rule Effectiveness

```python
# Group by exit reason
exit_analysis = (
    df.group_by("exit_reason")
    .agg([
        pl.count().alias("count"),
        pl.col("pnl").mean().alias("avg_pnl"),
        pl.col("return_pct").mean().alias("avg_return_pct")
    ])
)

# Analyze stop-loss effectiveness
stop_loss_trades = df.filter(pl.col("exit_reason") == "stop_loss")
```

### Context-Dependent Performance

```python
# Performance by VIX regime
vix_analysis = (
    df.with_columns([
        pl.when(pl.col("vix_entry") < 15)
        .then(pl.lit("low"))
        .when(pl.col("vix_entry") < 25)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("high"))
        .alias("vix_regime")
    ])
    .group_by("vix_regime")
    .agg([
        pl.count().alias("count"),
        pl.col("pnl").sum().alias("total_pnl"),
        pl.col("return_pct").mean().alias("avg_return")
    ])
)
```

### Feature Importance Analysis

```python
# Analyze which features correlate with profitable trades
profitable = df.filter(pl.col("pnl") > 0)
unprofitable = df.filter(pl.col("pnl") <= 0)

feature_stats = pl.DataFrame({
    "feature": ["atr_entry", "rsi_entry", "momentum_entry", "volatility_entry"],
    "profitable_mean": [
        profitable["atr_entry"].mean(),
        profitable["rsi_entry"].mean(),
        profitable["momentum_entry"].mean(),
        profitable["volatility_entry"].mean(),
    ],
    "unprofitable_mean": [
        unprofitable["atr_entry"].mean(),
        unprofitable["rsi_entry"].mean(),
        unprofitable["momentum_entry"].mean(),
        unprofitable["volatility_entry"].mean(),
    ]
})
```

## Design Rationale

### Why Polars-Native?

- **Performance**: 10-100x faster than pandas for large trade sets
- **Memory Efficiency**: Lazy evaluation and optimized memory layout
- **Type Safety**: Strong typing prevents data corruption
- **Arrow Compatibility**: Seamless interop with other tools

### Why Separate Entry/Exit Fields?

- **Analysis Flexibility**: Compare entry vs exit conditions
- **Feature Attribution**: Understand which features matter for entry vs exit
- **ML Model Evaluation**: Validate predictions against actual outcomes

### Why Comprehensive Context?

- **Strategy Debugging**: Understand why trades were taken
- **Risk Management Review**: Evaluate rule effectiveness
- **Performance Attribution**: Identify what drives returns
- **Model Improvement**: Discover blind spots in ML models

## Integration with TradeTracker

The existing `TradeRecord` in `trade_tracker.py` will continue to work for basic use cases. `MLTradeRecord` is an enhanced schema for strategies that:

- Use ML models for entry/exit decisions
- Track technical indicators and features
- Implement sophisticated risk management
- Need detailed post-backtest analysis

Both schemas can coexist. Users can choose the appropriate schema for their needs.

## Future Enhancements

Potential additions in future versions:

- Portfolio-level context (aggregate exposure, correlation)
- Multi-leg trade support (pairs, spreads)
- Execution quality metrics (VWAP deviation, fill quality)
- Custom field validation with Pydantic
- Schema versioning for backward compatibility
