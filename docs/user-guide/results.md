# Results & Analysis

`Engine.run()` returns a `BacktestResult` containing trades, equity curve, fills, and computed metrics. Everything is accessible as Python objects, Polars DataFrames, or Parquet files.

## Metrics

```python
result = engine.run()
m = result.metrics

# Returns
print(f"Total Return:  {m['total_return_pct']:.1f}%")
print(f"CAGR:          {m['cagr']:.2%}")
print(f"Volatility:    {m['volatility']:.2%}")

# Risk
print(f"Max Drawdown:  {m['max_drawdown_pct']:.1f}%")
print(f"Sharpe:        {m['sharpe']:.2f}")
print(f"Sortino:       {m['sortino']:.2f}")
print(f"Calmar:        {m['calmar']:.2f}")

# Trades
print(f"Trades:        {m['num_trades']}")
print(f"Win Rate:      {m['win_rate']:.1%}")
print(f"Profit Factor: {m['profit_factor']:.2f}")
print(f"Expectancy:    ${m['expectancy']:.2f}")
print(f"Avg Win:       ${m['avg_win']:.2f}")
print(f"Avg Loss:      ${m['avg_loss']:.2f}")

# Costs
print(f"Commission:    ${m['total_commission']:.2f}")
print(f"Slippage:      ${m['total_slippage']:.2f}")
```

### Available Metrics

| Metric | Description |
|--------|-------------|
| `initial_cash` | Starting cash |
| `final_value` | Final portfolio value |
| `total_return` | Total return as decimal |
| `total_return_pct` | Total return as percentage |
| `cagr` | Compound annual growth rate |
| `volatility` | Annualized volatility |
| `max_drawdown` | Maximum drawdown (positive number) |
| `max_drawdown_pct` | Maximum drawdown as percentage |
| `sharpe` | Sharpe ratio |
| `sortino` | Sortino ratio |
| `calmar` | Calmar ratio |
| `num_trades` | Total completed trades |
| `winning_trades` | Number of winning trades |
| `losing_trades` | Number of losing trades |
| `win_rate` | Win rate (0 to 1) |
| `profit_factor` | Gross profits / gross losses |
| `expectancy` | Average $ per trade |
| `avg_trade` | Average trade P&L |
| `avg_win` | Average winning trade |
| `avg_loss` | Average losing trade |
| `largest_win` | Largest single win |
| `largest_loss` | Largest single loss |
| `total_commission` | Total commission paid |
| `total_slippage` | Total slippage cost |
| `skipped_bars` | Bars skipped by calendar filter |

## Trades DataFrame

```python
trades_df = result.to_trades_dataframe()
print(trades_df.head())
```

Returns a Polars DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | String | Asset identifier |
| `entry_time` | Datetime | Entry timestamp |
| `exit_time` | Datetime | Exit timestamp |
| `entry_price` | Float | Entry fill price |
| `exit_price` | Float | Exit fill price |
| `quantity` | Float | Position size |
| `direction` | String | "long" or "short" |
| `pnl` | Float | Dollar P&L |
| `pnl_percent` | Float | Percentage return |
| `bars_held` | Int | Holding period |
| `fees` | Float | Total commission |
| `slippage` | Float | Total slippage |
| `mfe` | Float | Maximum favorable excursion |
| `mae` | Float | Maximum adverse excursion |
| `exit_reason` | String | Why the trade exited |
| `status` | String | "closed" or "open" |

Open positions at the end of the backtest are included with `status="open"` and mark-to-market values.

## Equity DataFrame

```python
equity_df = result.to_equity_dataframe()
print(equity_df.head())
```

Returns a Polars DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | Datetime | Bar timestamp |
| `equity` | Float | Portfolio value |
| `return` | Float | Bar-to-bar return |
| `cumulative_return` | Float | Cumulative return from start |
| `drawdown` | Float | Current drawdown from HWM |
| `high_water_mark` | Float | Running maximum equity |

## Dictionary Output

For backward compatibility:

```python
result_dict = result.to_dict()
# Same structure as result.metrics
```

## Parquet Export

Save results for later analysis or integration with ml4t-diagnostic:

```python
# Export trades and equity to Parquet
result.to_parquet("./results/my_backtest")
# Creates: my_backtest_trades.parquet, my_backtest_equity.parquet

# Reload later
from ml4t.backtest.result import BacktestResult
result = BacktestResult.from_parquet("./results/my_backtest")
```

## Integration with ml4t-diagnostic

Convert trades to TradeRecord format for the diagnostic library:

```python
# Bridge to ml4t-diagnostic
trade_records = result.to_trade_records()

# Or use the bridge function directly
from ml4t.backtest.analytics.bridge import to_trade_records
records = to_trade_records(result.trades)
```

## Fills

Access every individual order fill:

```python
for fill in result.fills:
    print(f"{fill.asset}: {fill.quantity} @ ${fill.price:.2f}")
    print(f"  Commission: ${fill.commission:.2f}")
    print(f"  Slippage: ${fill.slippage:.4f}")
```

## Config Preservation

The config used for the backtest is preserved in the result:

```python
print(result.config.describe())
print(result.config.preset_name)
```

## See It in Action

The [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) book uses BacktestResult in every case study:

- **Ch16 / NB05** (`performance_reporting`) — comprehensive metrics extraction, equity curve visualization, trade analysis
- **Ch16 case studies** — all cases call `result.to_daily_returns(calendar="NYSE")` for integration with ml4t-diagnostic signal analysis
- **Ch16 / NB06** (`sharpe_ratio_inference`) — statistical inference on backtest results

## Next Steps

- [Quickstart](../getting-started/quickstart.md) -- end-to-end examples
- [Profiles](profiles.md) -- compare results across framework profiles
