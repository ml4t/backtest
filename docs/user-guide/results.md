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

# Per-trade returns (percentage-based, direction-aware)
print(f"Avg Trade:     {m['avg_trade']:.2%}")
print(f"Avg Win:       {m['avg_win']:.2%}")
print(f"Avg Loss:      {m['avg_loss']:.2%}")
print(f"Best Trade:    {m['largest_win']:.2%}")
print(f"Worst Trade:   {m['largest_loss']:.2%}")
print(f"Payoff Ratio:  {m['payoff_ratio']:.2f}")

# Costs
print(f"Commission:    ${m['total_commission']:.2f}")
print(f"Slippage:      ${m['total_slippage']:.2f}")
print(f"Total Costs:   ${m['total_costs']:.2f}")
print(f"Avg Cost Drag: {m['avg_cost_drag']:.4%}")

# Gross vs Net
print(f"Gross P&L:     ${m['total_gross_pnl']:.2f}")
print(f"Gross PF:      {m['gross_profit_factor']:.2f}")
print(f"Net PF:        {m['profit_factor']:.2f}")
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
| `profit_factor` | Net profit factor (winning P&L / losing P&L) |
| `expectancy` | Expected return per trade (decimal) |
| `avg_trade` | Average trade return (decimal) |
| `avg_win` | Average winning trade return (decimal) |
| `avg_loss` | Average losing trade return (decimal, negative) |
| `largest_win` | Best single trade return (decimal) |
| `largest_loss` | Worst single trade return (decimal, negative) |
| `payoff_ratio` | avg_win / \|avg_loss\| (size-normalized reward-to-risk) |
| `total_commission` | Total commission paid |
| `total_slippage` | Total slippage cost (entry + exit) |
| `total_gross_pnl` | Total P&L from price moves only (before costs) |
| `total_costs` | Total transaction costs (commission + slippage) |
| `avg_cost_drag` | Average cost as fraction of trade notional |
| `gross_profit_factor` | Profit factor from raw price moves (isolates edge from costs) |
| `skipped_bars` | Bars skipped by calendar filter |

### Cost Decomposition

Every trade carries a full cost breakdown, letting you separate strategy edge from execution costs:

```python
for trade in result.trades:
    print(f"{trade.symbol}: gross={trade.gross_pnl:+.2f}, "
          f"net={trade.pnl:+.2f}, drag={trade.cost_drag:.4%}")
```

| Property | Description |
|----------|-------------|
| `trade.gross_pnl` | Price-move P&L: `(exit - entry) * qty * multiplier` |
| `trade.pnl` | Net P&L after all costs |
| `trade.gross_return` | Direction-aware gross return (same as `pnl_percent`) |
| `trade.net_return` | Direction-aware net return including fees |
| `trade.total_slippage_cost` | Entry + exit slippage in dollars |
| `trade.cost_drag` | Total cost as fraction of notional |
| `trade.fees` | Total commission (entry + exit) |
| `trade.entry_slippage` | Per-unit slippage on entry |
| `trade.slippage` | Per-unit slippage on exit |
| `trade.multiplier` | Contract multiplier (1.0 for equities, 50.0 for ES futures) |

`pnl_percent` is direction-aware: positive means profitable for both long and short trades.

## Trade Analyzer

`result.trade_analyzer` provides aggregate statistics on closed trades:

```python
ta = result.trade_analyzer

# Standard metrics
print(f"Win Rate:           {ta.win_rate:.1%}")
print(f"Profit Factor:      {ta.profit_factor:.2f}")
print(f"Avg MFE:            {ta.avg_mfe:.4f}")
print(f"MFE Capture:        {ta.mfe_capture_ratio:.2f}")

# Cost decomposition
print(f"Gross P&L:          ${ta.total_gross_pnl:.2f}")
print(f"Net Profit:         ${ta.net_profit:.2f}")
print(f"Total Costs:        ${ta.total_costs:.2f}")
print(f"Avg Cost Drag:      {ta.avg_cost_drag:.4%}")
print(f"Gross Profit Factor:{ta.gross_profit_factor:.2f}")

# Filter by side
long_stats = ta.by_side("long")
short_stats = ta.by_side("short")
print(f"Long win rate: {long_stats.win_rate:.1%}")
print(f"Short win rate: {short_stats.win_rate:.1%}")

# Export all stats
stats_dict = ta.to_dict()
```

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
| `quantity` | Float | Position size (negative for shorts) |
| `direction` | String | "long" or "short" |
| `pnl` | Float | Net P&L after costs |
| `pnl_percent` | Float | Direction-aware percentage return |
| `bars_held` | Int | Holding period |
| `fees` | Float | Total commission |
| `slippage` | Float | Exit slippage |
| `mfe` | Float | Maximum favorable excursion |
| `mae` | Float | Maximum adverse excursion |
| `entry_slippage` | Float | Per-unit slippage on entry |
| `multiplier` | Float | Contract multiplier (futures) |
| `gross_pnl` | Float | Price-move P&L before fees |
| `net_return` | Float | Direction-aware net return including fees |
| `total_slippage_cost` | Float | Entry + exit slippage in dollars |
| `cost_drag` | Float | Total cost as fraction of notional |
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

## Fills

Access every individual order fill:

```python
for fill in result.fills:
    print(f"{fill.asset}: {fill.quantity} @ ${fill.price:.2f}")
    print(f"  Type: {fill.order_type}")
    print(f"  Commission: ${fill.commission:.2f}")
    print(f"  Slippage: ${fill.slippage:.4f}")
```

Fill objects carry order-type metadata for audit:

| Field | Description |
|-------|-------------|
| `fill.order_type` | `"market"`, `"limit"`, or `"stop"` |
| `fill.limit_price` | Limit price (for limit orders) |
| `fill.stop_price` | Stop price (for stop orders) |
| `fill.price` | Actual fill price |
| `fill.commission` | Commission charged |
| `fill.slippage` | Slippage applied |

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

### Portfolio Analysis (Recommended)

The simplest way to bridge backtest results into ml4t-diagnostic is `to_portfolio_analysis()`:

```python
from ml4t.backtest import Engine

result = engine.run()

# One-liner bridge to ml4t-diagnostic
analysis = result.to_portfolio_analysis(calendar="NYSE")

# Now use PortfolioAnalysis methods
print(f"Sharpe: {analysis.sharpe_ratio():.2f}")
print(f"Max DD: {analysis.max_drawdown():.2%}")
monthly = analysis.compute_monthly_returns()
```

The method extracts daily returns via `to_daily_pnl()` and sets `periods_per_year` from the calendar (252 for NYSE, 365 for crypto, etc.). If no calendar is passed, it uses the config's calendar.

```python
# Crypto backtest
analysis = result.to_portfolio_analysis(calendar="crypto")

# With benchmark
analysis = result.to_portfolio_analysis(
    calendar="NYSE",
    benchmark=spy_returns,  # numpy array or Polars Series
)

# Gross vs net comparison
analysis_gross = results_gross.to_portfolio_analysis(calendar="crypto")
analysis_net = results_net.to_portfolio_analysis(calendar="crypto")
```

!!! note "Requires ml4t-diagnostic"
    Install with `pip install ml4t-diagnostic`. The import is deferred so ml4t-backtest
    works standalone without ml4t-diagnostic installed.

### Trade Records

Convert trades to TradeRecord format for the diagnostic library:

```python
# Bridge to ml4t-diagnostic
trade_records = result.to_trade_records()

# Or use the bridge function directly
from ml4t.backtest.analytics.bridge import to_trade_records
records = to_trade_records(result.trades)
```

The bridge exports all cost decomposition fields (`gross_pnl`, `net_return`, `total_slippage_cost`, `cost_drag`) for diagnostic analysis.

### Full Tearsheet

Pass all result data for the richest tearsheet (up to 24 sections):

```python
from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

html = generate_backtest_tearsheet(
    trades=result.to_trades_dataframe(),
    returns=analysis.returns,
    equity_curve=result.to_equity_dataframe(),
    metrics=result.metrics,
    template="full",
    title="My Strategy — Full Report",
    output_path="tearsheet.html",
)
```

#### Metrics Keys That Enable Tearsheet Sections

The `metrics` dict controls which tearsheet sections render. Sections gracefully degrade when keys are missing.

| Section | Required Metrics Keys |
|---------|----------------------|
| Executive Summary | `sharpe_ratio`, `max_drawdown`, `win_rate`, `profit_factor`, `n_trades`, `cagr`, `volatility`, `expectancy` |
| Cost Attribution | `gross_pnl`, `commission`, `slippage` |
| Statistical Validity (DSR) | `dsr_probability`, `dsr_significant`, `min_trl`, `current_trl`, `trl_sufficient` |
| RAS Adjustment | `ras_adjusted_ic`, `ras_significant`, `original_ic`, `rademacher_complexity` |
| Confidence Intervals | `sharpe_ratio`, `sharpe_ratio_lower_95`, `sharpe_ratio_upper_95` (similarly for other metrics) |
| Haircut Sharpe | `sharpe`, `n_periods` or `n_observations` |
| Expected Max Sharpe | `expected_max_sharpe` |

Sections that depend only on `trades` (trade analysis, MFE/MAE, exit reasons) or `returns` (drawdown, monthly heatmap, rolling Sharpe) require no special metrics keys.

## Config Preservation

The config used for the backtest is preserved in the result:

```python
print(result.config.describe())
print(result.config.preset_name)
```

## See It in Action

The [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) book uses BacktestResult in every case study:

- **Ch16 / NB05** (`performance_reporting`) — `to_portfolio_analysis()`, MFE/MAE analysis, gross vs net comparison, full 24-section tearsheet
- **Ch16 case studies** — all cases save trade artifacts via `to_parquet()` and pass trades/metrics/equity to tearsheet generation
- **Ch16 / NB06** (`sharpe_ratio_inference`) — statistical inference on backtest results

## Next Steps

- [Quickstart](../getting-started/quickstart.md) -- end-to-end examples
- [Profiles](profiles.md) -- compare results across framework profiles
