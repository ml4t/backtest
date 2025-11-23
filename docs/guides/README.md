# ml4t.backtest User Guides

## Getting Started

- **[Quick Start](./QUICKSTART.md)** - Get up and running in 5 minutes

## Migration

- **[Migration Guide](./MIGRATION_GUIDE.md)** - Migrate from Zipline or Backtrader
  - Single-asset and multi-asset examples
  - Pre-computed signals workflow
  - Key API differences

## Advanced Topics

- **[Differentiators](./DIFFERENTIATORS.md)** - What ml4t.backtest does better
  - Time-based exits (`TimeExit`)
  - Dynamic stop tightening
  - ML confidence → position sizing
  - Cash vs margin accounts
  - Futures with multipliers
  - Position-aware exit rules

## Reference

- **[Data Feeds](./data_feeds.md)** - DataFeed API and data format
- **[Risk Management](./risk_management_quickstart.md)** - Position rules and portfolio limits
- **[Configuration](../configuration_guide.md)** - Engine configuration options

## Philosophy

ml4t.backtest separates concerns:

1. **Data Preparation** - Load OHLCV, compute indicators/signals with Polars/pandas
2. **Backtesting** - Pure execution logic, no indicator computation
3. **Analysis** - Evaluate results with analytics module

This differs from Zipline/Backtrader where indicators are computed on-the-fly during the backtest. Our approach provides:

- **Speed** - Compute expensive indicators once
- **Reproducibility** - Same signals = same results
- **ML Integration** - Natural fit for ML model predictions
- **Simpler Strategies** - Focus on trading logic, not indicator boilerplate
