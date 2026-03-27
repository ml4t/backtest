# Results & Analysis

`Engine.run()` returns a `BacktestResult` containing trades, equity curve, fills,
portfolio state, and computed metrics. Everything is accessible as Python objects,
Polars DataFrames, or Parquet files.

This applies to both classic OHLCV backtests and quote-aware backtests. When you run
with bid/ask-aware execution or marking, the result surface preserves the quote
context used to produce fills and trade summaries.

For reproducibility, `BacktestResult` also exposes:

- `result.config.to_dict()` for the fully resolved replayable config payload
- `result.to_spec_dict()` for a richer runtime snapshot including library version and realized window
- `result.to_predictions_dataframe()` for the raw prediction/input surface passed into the
  backtest, when available
- `result.to_parquet(...)`, which writes `config.yaml`, `spec.yaml`, and `predictions.parquet`
  when available

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
print(f"Fills:         {m['num_fills']}")
print(f"Rebalances:    {m['num_rebalance_events']}")
print(f"Symbols:       {m['unique_symbols_traded']}")
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
print(f"Filled Notional: ${m['total_filled_notional']:,.2f}")
print(f"Avg Turnover:    {m['avg_turnover']:.2%}")
print(f"Max Turnover:    {m['max_turnover']:.2%}")
print(f"Avg Open Pos:    {m['avg_open_positions']:.2f}")
print(f"Max Open Pos:    {m['max_open_positions']}")

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
| `num_fills` | Total execution events |
| `num_rebalance_events` | Unique timestamps with at least one fill |
| `unique_symbols_traded` | Number of symbols with at least one fill |
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
| `total_slippage` | Total slippage cost in dollars (entry + exit) |
| `total_filled_notional` | Sum of absolute filled notional across all fills |
| `avg_turnover` | Mean per-bar one-way turnover from fills |
| `max_turnover` | Maximum per-bar one-way turnover from fills |
| `avg_open_positions` | Mean number of open positions across bars |
| `max_open_positions` | Maximum number of open positions across bars |
| `total_gross_pnl` | Total P&L from price moves only (before costs) |
| `total_costs` | Total transaction costs (commission + slippage) |
| `avg_cost_drag` | Average cost as fraction of trade notional |
| `gross_profit_factor` | Profit factor from raw price moves (isolates edge from costs) |
| `skipped_bars` | Bars skipped by calendar filter |

### Reporting Model

`BacktestResult` now exposes three distinct raw reporting surfaces:

- `trades`: flat-to-flat lifecycle summaries
- `fills`: execution blotter rows
- `portfolio_state`: end-of-bar portfolio snapshots

For rebalancing strategies, `num_trades` is not the right proxy for trading activity.
Use `num_fills`, `num_rebalance_events`, and `total_filled_notional` instead.

Turnover uses a one-way execution-based definition:

```python
turnover_t = filled_notional_at_timestamp / equity_at_timestamp
```

Bars with no fills contribute `0`. This means:

- buying a fully cash portfolio into a fully invested book is turnover `1.0`
- selling a fully invested book back to cash is turnover `1.0`
- fully rotating one full book into another is turnover `2.0`

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
| `trade.exit_slippage` | Per-unit slippage on exit |
| `trade.multiplier` | Contract multiplier (1.0 for equities, 50.0 for ES futures) |

Trade records also summarize nullable quote context for the entry and exit:

| Property | Description |
|----------|-------------|
| `trade.entry_quote_mid_price` / `trade.exit_quote_mid_price` | Quote midpoint at fill time |
| `trade.entry_bid_price` / `trade.exit_bid_price` | Best bid at fill time |
| `trade.entry_ask_price` / `trade.exit_ask_price` | Best ask at fill time |
| `trade.entry_spread` / `trade.exit_spread` | Bid/ask spread at fill time |
| `trade.entry_available_size` / `trade.exit_available_size` | Side-aware available quote size |

`pnl_percent` is direction-aware: positive means profitable for both long and short trades.

Quote-aware backtests therefore leave an explicit audit trail:

- fills record the reference price, bid, ask, midpoint, spread, and available size
- trades summarize entry/exit quote context
- portfolio state reflects the configured `mark_price`

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
| `exit_slippage` | Float | Exit slippage |
| `mfe` | Float | Maximum favorable excursion |
| `mae` | Float | Maximum adverse excursion |
| `entry_slippage` | Float | Per-unit slippage on entry |
| `multiplier` | Float | Contract multiplier (futures) |
| `entry_quote_mid_price` | Float | Entry quote midpoint |
| `entry_bid_price` | Float | Entry best bid |
| `entry_ask_price` | Float | Entry best ask |
| `entry_spread` | Float | Entry spread |
| `entry_available_size` | Float | Entry-side available size |
| `exit_quote_mid_price` | Float | Exit quote midpoint |
| `exit_bid_price` | Float | Exit best bid |
| `exit_ask_price` | Float | Exit best ask |
| `exit_spread` | Float | Exit spread |
| `exit_available_size` | Float | Exit-side available size |
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

## Portfolio State DataFrame

```python
portfolio_df = result.to_portfolio_state_dataframe()
print(portfolio_df.head())
```

Returns a Polars DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | Datetime | Bar timestamp |
| `equity` | Float | Portfolio equity after bar processing |
| `cash` | Float | Cash balance |
| `gross_exposure` | Float | Sum of absolute marked position values |
| `net_exposure` | Float | Signed sum of marked position values |
| `open_positions` | Int | Number of open positions |

This is the right primitive for downstream diagnostics such as:

- average invested fraction
- gross and net exposure analysis
- time in market
- occupancy and utilization metrics

## Predictions DataFrame

If you passed `signals_df` into `DataFeed` or `run_backtest(...)`, the raw input
is available on the result:

```python
predictions_df = result.to_predictions_dataframe()
print(predictions_df.head())
```

This returns the original Polars DataFrame without reinterpretation. It is meant
for downstream diagnostics and attribution, for example:

- joining predictions to fills or trades
- evaluating score distributions for winning vs losing trades
- comparing model inputs to realized execution and PnL

The important convention is that this surface is treated as the raw model/input
surface, not as a guaranteed post-mapping trading signal surface.

## Fills

Access every individual order fill:

```python
for fill in result.fills:
    print(f"{fill.asset}: {fill.quantity} @ ${fill.price:.2f}")
    print(f"  Type: {fill.order_type}")
    print(f"  Commission: ${fill.commission:.2f}")
    print(f"  Slippage: ${fill.slippage:.4f}")
```

Or convert them directly to a Polars DataFrame:

```python
fills_df = result.to_fills_dataframe()
print(fills_df.select(["asset", "side", "price", "price_source", "bid_price", "ask_price"]))
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
| `fill.price_source` | Configured source used for the fill |
| `fill.reference_price` | Feed reference price (`bar["price"]`) |
| `fill.quote_mid_price` | Quote midpoint at fill time |
| `fill.bid_price` / `fill.ask_price` | Best bid / ask |
| `fill.spread` | Bid-ask spread |
| `fill.bid_size` / `fill.ask_size` | Quote sizes |
| `fill.available_size` | Side-aware size used for the fill context |

For quote-aware backtests, `fills.parquet` is the first place to look when you want
to verify whether a result difference came from:

- quote-side execution
- synthetic slippage
- commission
- the configured mark source

## Dictionary Output

For backward compatibility:

```python
result_dict = result.to_dict()
# Same structure as result.metrics
```

## Parquet Export

Save results for later analysis or integration with ml4t-diagnostic:

```python
# Export all result components to Parquet / JSON / YAML
result.to_parquet("./results/my_backtest")
# Creates:
#   trades.parquet
#   fills.parquet
#   predictions.parquet  # if raw prediction inputs were supplied
#   equity.parquet
#   portfolio_state.parquet
#   daily_pnl.parquet
#   metrics.json
#   config.yaml  # when config is attached

# Reload later
from ml4t.backtest.result import BacktestResult
result = BacktestResult.from_parquet("./results/my_backtest")
```

## Integration with ml4t-diagnostic

### Portfolio Analysis (Recommended)

The simplest way to bridge backtest results into ml4t-diagnostic is
`portfolio_analysis_from_result()`:

```python
from ml4t.backtest import Engine
from ml4t.diagnostic.integration import portfolio_analysis_from_result

result = engine.run()

# One-liner bridge to ml4t-diagnostic
analysis = portfolio_analysis_from_result(result, calendar="NYSE")

# Now use PortfolioAnalysis methods
print(f"Sharpe: {analysis.sharpe_ratio():.2f}")
print(f"Max DD: {analysis.max_drawdown():.2%}")
monthly = analysis.compute_monthly_returns()
```

The helper extracts daily returns via `to_daily_pnl()` and sets
`periods_per_year` from the calendar (252 for NYSE, 365 for crypto, etc.).
If no calendar is passed, it uses the config's calendar.

For richer diagnostics, pass these alongside returns:

- `result.to_trades_dataframe()`
- `result.to_fills_dataframe()`
- `result.to_portfolio_state_dataframe()`

```python
# Crypto backtest
analysis = portfolio_analysis_from_result(result, calendar="crypto")

# With benchmark
analysis = portfolio_analysis_from_result(
    result,
    calendar="NYSE",
    benchmark=spy_returns,  # numpy array or Polars Series
)

# Gross vs net comparison
analysis_gross = portfolio_analysis_from_result(results_gross, calendar="crypto")
analysis_net = portfolio_analysis_from_result(results_net, calendar="crypto")
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

- **Ch16 / NB05** (`performance_reporting`) — `portfolio_analysis_from_result()`, MFE/MAE analysis, gross vs net comparison, full 24-section tearsheet
- **Ch16 case studies** — all cases save trade artifacts via `to_parquet()` and pass trades/metrics/equity to tearsheet generation
- **Ch16 / NB06** (`sharpe_ratio_inference`) — statistical inference on backtest results

## Next Steps

- [Book Guide](../book-guide/index.md) -- notebook and case-study map for reporting workflows
- [Quickstart](../getting-started/quickstart.md) -- end-to-end examples
- [Profiles](profiles.md) -- compare results across framework profiles
