# analytics/ - 951 Lines

Performance metrics and trade analysis.

## Modules

| File | Purpose |
|------|---------|
| metrics.py | Performance metrics (Sharpe, CAGR, drawdown) |
| equity.py | Equity curve analysis |
| trades.py | Trade statistics |
| bridge.py | ml4t-diagnostic integration bridge |

## Key Functions

`calculate_metrics()`, `EquityCurve`, `TradeAnalysis`

## ml4t-diagnostic Integration

The `bridge.py` module provides conversion functions for ml4t-diagnostic analysis.

### Bridge Functions

| Function | Purpose |
|----------|---------|
| `to_trade_record(trade)` | Convert single Trade to diagnostic TradeRecord format |
| `to_trade_records(trades)` | Batch convert list of trades |
| `to_returns_series(equity)` | Convert equity curve to returns for Sharpe analysis |
| `to_equity_dataframe(equity, timestamps)` | Convert equity history with timestamps |

### Usage Example

```python
from ml4t.backtest.analytics.bridge import to_trade_records, to_returns_series

# After running backtest
results = engine.run()

# Convert for diagnostic analysis
trade_records = to_trade_records(results["trades"])
returns = to_returns_series(results["equity_curve"])

# Use with ml4t.diagnostic
from ml4t.diagnostic.integration import TradeRecord
from ml4t.diagnostic.evaluation import sharpe_ratio

records = [TradeRecord(**r) for r in trade_records]
sr = sharpe_ratio(returns, confidence_intervals=True)
```

### Schema Alignment

Field names are aligned between backtest Trade and diagnostic TradeRecord (v0.1.0a6+):
- Direct mapping: symbol, entry_time, exit_time, entry_price, exit_price, pnl, fees, slippage
- Computed: duration (exit_time - entry_time), direction (from quantity sign)
- Aliases: timestamp (exit_time), entry_timestamp (entry_time)
