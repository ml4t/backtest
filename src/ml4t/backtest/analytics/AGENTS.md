# analytics/ - ~970 Lines

Performance metrics, trade analysis, cost decomposition, and ml4t-diagnostic integration.

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| trades.py | ~510 | Trade statistics (win rate, PnL, MFE/MAE, cost decomposition) |
| metrics.py | 180 | Performance metrics (Sharpe, CAGR, drawdown) |
| bridge.py | ~155 | ml4t-diagnostic integration bridge |
| equity.py | 119 | Equity curve calculation |

## Key Functions

`calculate_metrics()`, `to_trade_records()`, `to_returns_series()`

## TradeAnalyzer Cost Decomposition (v0.1.0b2)

`TradeAnalyzer` exposes aggregate cost decomposition metrics:
- `total_gross_pnl` - Price-move P&L before all costs
- `total_costs` - Total fees + slippage
- `avg_cost_drag` - Average cost as fraction of notional
- `gross_profit_factor` - Profit factor from raw price moves (isolates edge from costs)

## ml4t-diagnostic Bridge

`bridge.py` converts backtest Trade objects to diagnostic TradeRecord format:
- `to_trade_record(trade)` / `to_trade_records(trades)` - Trade conversion
- `to_returns_series(equity)` - Equity to returns for Sharpe analysis
- `to_equity_dataframe(equity, timestamps)` - Equity with timestamps

Bridge exports cost decomposition fields: `gross_pnl`, `net_return`, `total_slippage_cost`, `cost_drag`
