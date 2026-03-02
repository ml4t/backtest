# analytics/ - 917 Lines

Performance metrics, trade analysis, and ml4t-diagnostic integration.

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| trades.py | 472 | Trade statistics (win rate, PnL, MFE/MAE) |
| metrics.py | 180 | Performance metrics (Sharpe, CAGR, drawdown) |
| bridge.py | 146 | ml4t-diagnostic integration bridge |
| equity.py | 119 | Equity curve calculation |

## Key Functions

`calculate_metrics()`, `to_trade_records()`, `to_returns_series()`

## ml4t-diagnostic Bridge

`bridge.py` converts backtest Trade objects to diagnostic TradeRecord format:
- `to_trade_record(trade)` / `to_trade_records(trades)` - Trade conversion
- `to_returns_series(equity)` - Equity to returns for Sharpe analysis
- `to_equity_dataframe(equity, timestamps)` - Equity with timestamps
