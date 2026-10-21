# Results & Analysis

The [result export tutorial](../tutorials/results-and-analysis.md) joins a
time-zone-aware feed to fills, equity, predictions, and funding, then checks a
Parquet round trip. The [diagnostic handoff](../tutorials/diagnostic-handoff.md)
uses the optional analysis package.

`Engine.run()` returns a `BacktestResult` containing trades, equity curve, fills,
portfolio state, and computed metrics. Everything is accessible as Python objects,
Polars DataFrames, or Parquet files.

The `trades`, `fills`, `equity_curve`, `portfolio_state`, and `rejected_orders` lists are
snapshots: adding to or removing from engine or broker state after `run()` does not change a
completed result.

This applies to both classic OHLCV backtests and quote-aware backtests. When you run
with bid/ask-aware execution or marking, the result surface preserves the quote
context used to produce fills and trade summaries.

For reproducibility, `BacktestResult` also exposes:

- `result.config.to_dict()` for the fully resolved replayable config payload
- `result.to_spec_dict()` for a richer runtime snapshot including library version and realized window
- `result.to_predictions_dataframe()` for the raw prediction/input surface passed into the
  backtest, when available
- `result.rejected_orders` and `result.to_rejected_orders_dataframe()` for orders that reached
  the rejected terminal state
- `result.to_parquet(...)`, which writes `config.yaml`, `spec.yaml`, and `predictions.parquet`
  when available

Snippets using `result` assume an existing `BacktestResult` from `Engine.run()`. The linked export tutorial supplies the complete run and checks each exported table.

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
| `num_trades` | Realized exit legs, including partial reductions and full closes |
| `num_orders` | Total submitted orders |
| `num_rejected_orders` | Orders that reached the rejected terminal state |
| `num_fills` | Total execution events |
| `num_rebalance_events` | Unique timestamps with at least one fill |
| `unique_symbols_traded` | Number of symbols with at least one fill |
| `winning_trades` | Winning realized exit legs |
| `losing_trades` | Losing realized exit legs |
| `win_rate` | Winning fraction across realized exit legs (0 to 1) |
| `profit_factor` | Winning P&L / losing P&L across realized exit legs |
| `expectancy` | Expected return per realized exit leg (decimal) |
| `avg_trade` | Average realized exit-leg return (decimal) |
| `avg_win` | Average winning realized exit-leg return (decimal) |
| `avg_loss` | Average losing realized exit-leg return (decimal, negative) |
| `largest_win` | Best realized exit-leg return (decimal) |
| `largest_loss` | Worst realized exit-leg return (decimal, negative) |
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

- `trades`: realized exit legs plus end-of-backtest open-position marks
- `fills`: execution blotter rows
- `portfolio_state`: account observations, with end-of-bar snapshots by default

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

`result.trade_analyzer` computes P&L statistics from realized exit legs, including partial
reductions. Holding-period and MAE/MFE statistics use fully closed position lifecycles. Those
lifecycle values are `NaN` when partial realizations exist but no position has fully closed.

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
| `exit_reason_detail` | String | Detailed risk or liquidation cause, when available |
| `status` | String | "closed", "partial", or "open" |

Partial reductions use `status="partial"`. Realized-P&L metrics include partial and fully closed
records, while holding-period and excursion metrics use only fully closed records to avoid counting
one position lifecycle more than once. Open positions at the end of the backtest use `status="open"`
and mark-to-market values.

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
| `return` | Float | Return from the preceding NAV, or recorded initial capital for the first observation |
| `cumulative_return` | Float | Cumulative return from start |
| `drawdown` | Float | Current drawdown from HWM |
| `high_water_mark` | Float | Running maximum investment wealth, including recorded initial capital |
| `performance_equity` | Float | Investment wealth at initial capital scale, excluding external flows |

The capital-flow, recorded-variable and detailed holdings interfaces below are
part of the upcoming successor release. Earlier artifacts remain readable with
their recorded component availability and valuation baseline.

## External capital and investment returns

Pass a prepared `capital_flows` Polars table to `DataFeed`, or to `run_backtest`.
Each row requires `timestamp`, unique `event_id`, `kind` (`"delta"` or `"target"`)
and finite `value` in account cash units. Negative deltas withdraw cash; targets
specify total marked NAV, not a cash-balance target. Optional `known_at` cannot be
later than the application timestamp. Each application must match an accepted feed
observation. Every held asset must have a finite current quote for the configured
mark method at that instant. A stale carried quote cannot establish the pre-flow
valuation. All-cash changes need no asset quote; zero is a valid current valuation
mark for an already-held asset. No exchange rate or missing event valuation is inferred.

The engine marks current positions, applies external capital, then processes that
bar's funding, fills and decisions. The change is visible to that decision's
buying-power checks. Same-time rows retain input order, and the complete group is
validated before cash changes. A withdrawal cannot create negative cash in an
unleveraged account. Funding, fees and trading P&L remain investment returns.

`to_capital_flows_dataframe()` records the request, actual cash delta and NAV
immediately before and after application. Returns chain the independently valued
subperiods around each flow. A contribution does not become investment profit.
`equity` remains reported account NAV; `performance_equity` scales compounded
investment returns to initial capital. Drawdown and `high_water_mark` use that
investment wealth, so contributions and withdrawals do not create drawdowns.

<!-- ml4t-doc-test: results-external-capital -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode

class Invest(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if not broker.positions:
            broker.submit_order("TEST", 100)

dates = [datetime(2025, 1, day, 16, tzinfo=UTC) for day in (2, 3, 6)]
prices = pl.DataFrame({
    "timestamp": dates, "asset": ["TEST"] * 3, "close": [100.0, 110.0, 98.0]
})
flows = pl.DataFrame({
    "timestamp": [dates[1]], "event_id": ["deposit"],
    "kind": ["target"], "value": [12000.0]
})
result = Engine(
    DataFeed(prices_df=prices, capital_flows=flows), Invest(),
    BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
    holdings_sampling="bar",
).run()
assert result.to_capital_flows_dataframe()["cash_delta"].to_list() == [1000.0]
assert result.to_portfolio_state_dataframe()["equity"].to_list() == [10000.0, 12000.0, 10800.0]
print(f"Investment return: {result.metrics['total_return_pct']:.1f}%")
```

```text
Investment return: -1.0%
```

## Recorded strategy values

Call `broker.record(score=value, ...)` from a strategy to retain non-account
values. The latest value carries into later accepted bars until replaced.
`result.to_recorded_records()` returns decoded values grouped by timestamp and
zero-based return-bar index. Names stay inside each row's `values` mapping:
recording a variable called `cash` does not change reported account cash.

Recording copies values when called and samples them at bar end. JSON-compatible
values, NumPy arrays/scalars and Polars Series use the existing portable artifact
conversion; non-finite floats retain their tagged representation on disk.
Unsupported Python objects and reserved artifact-tag keys raise `ValueError`
before any value changes. If recording was never requested, the component is
unavailable rather than an empty history of zero signals.

<!-- ml4t-doc-test: results-recorded-values -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

class Record(Strategy):
    def on_start(self, broker):
        broker.record(score=1.0, cash=42)

    def on_data(self, timestamp, data, context, broker):
        if timestamp.day == 3:
            broker.record(score=2.0)

prices = pl.DataFrame({
    "timestamp": [datetime(2025, 1, day, 16, tzinfo=UTC) for day in (2, 3)],
    "asset": ["TEST", "TEST"], "close": [100.0, 101.0],
})
result = Engine(DataFeed(prices_df=prices), Record(), BacktestConfig(calendar=None)).run()
assert [row["values"]["score"] for row in result.to_recorded_records()] == [1.0, 2.0]
assert result.to_portfolio_state_dataframe()["cash"].to_list() == [100000.0, 100000.0]
```

`to_recorded_dataframe()` exposes a long table with `timestamp`, `bar_index`,
`name` and portable `value_json`. The existing Parquet artifact stores this
optional component and its bar-end sampling declaration. The decoded record view
restores the same values after strict reload.

## Observed holdings

Enable per-asset capture explicitly when creating the engine:

```python
result = Engine(feed, strategy, config, holdings_sampling="bar").run()
holdings = result.to_holdings_dataframe()
account = result.to_portfolio_state_dataframe()
```

Each holdings row records `timestamp`, `observation_id`, `asset`, signed
`quantity`, `mark`, `multiplier`, declared `currency`, `signed_notional` and
`liquidation_value`, `mark_source` and `mark_timestamp`. The latter retains the
original observation time when a last known mark is carried. `last_observed`
means the last finite reference price; unusable current quotes stay unavailable
for execution and do not replace that mark. A valid zero mark remains zero. These are observed
simulator positions, including open lots,
not positions reconstructed from closed trades. Currency is null when no contract
specification declares it. The current signed-notional ledger reports identical
notional and liquidation values; another account method must define them separately.
`result.account_method` and the artifact manifest explicitly declare
`"signed_notional"` for current Engine runs. Legacy artifacts without a declaration
report an unknown method; margin settings alone do not imply variation cash.

Account rows retain every accepted bar, including a completely flat portfolio.
Join on `observation_id` to distinguish a complete all-cash observation from a
missing observation. Capture defaults to `"none"`; asking for uncaptured holdings
raises `ArtifactError`. An empty captured table means a complete all-cash run.
Available sampling methods are:

- `"none"`: retain ordinary account bars without holdings capture.
- `"bar"`: capture complete holdings after every accepted bar.
- `"session"`: capture the last delivered bar in each declared calendar session.
- `"fill_and_session"`: also capture account and holdings immediately after each fill.

Session methods require a calendar. A last delivered bar is not an invented
exchange-close observation. Post-fill snapshots use the configured bar mark
policy, rather than fabricating an intrabar market-price path. Feed timestamp
semantics apply to both observation and mark timestamps.

Account `observation_kind` distinguishes `after_fill`, `bar` and
`session_last_observed`. Select `holdings_complete=True` for the complete capture
keys; rows without capture are not flat portfolios. Account `fill_sequence`
identifies the corresponding execution for a post-fill observation. A buy/sell
round trip can therefore retain its temporary holdings even if every bar ends
flat. Investment returns still use the original bar valuation timeline.

The existing Parquet export stores the optional holdings component and its sampling
declaration. Strict reload checks account keys and fill sequence. Old artifacts
without capture remain readable and report holdings unavailable.

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

`observation_id` identifies each account snapshot. Fill exports also include a
zero-based `fill_sequence` that preserves execution order at equal timestamps.
Fill `signed_quantity`, `multiplier` and declared `currency` retain execution
units. Legacy fills without those declarations keep null multiplier/currency,
rather than inferring them from a later contract specification. The manifest
records whether investment returns include recorded commissions; it remains
unknown for older artifacts without that declaration.

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
| `fill.order_type` | Native order type, including `"stop_limit"`, `"moc"` and `"trailing_stop"` |
| `fill.limit_price` | Limit price (for limit orders) |
| `fill.stop_price` | Stop price (for stop orders) |
| `fill.stop_triggered_at` | Observed activation instant, retained across later partial fills; None when unavailable |
| `fill.price` | Actual fill price |
| `fill.commission` | Commission charged |
| `fill.slippage` | Slippage applied |
| `fill.price_source` | Resolved configured execution source, including a next-bar override; conditional order fields identify level-based pricing |
| `fill.reference_price` | Feed reference price (`bar["price"]`) |
| `fill.quote_mid_price` | Quote midpoint at fill time |
| `fill.bid_price` / `fill.ask_price` | Best bid / ask |
| `fill.spread` | Bid-ask spread |
| `fill.bid_size` / `fill.ask_size` | Quote sizes |
| `fill.available_size` | Side-aware size used for the fill context |
| `fill.exit_reason` | Typed exit category; empty for entry fills |
| `fill.exit_reason_detail` | Detailed risk or liquidation cause, when available |

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
#   rejected_orders.parquet
#   predictions.parquet  # if raw prediction inputs were supplied
#   equity.parquet
#   portfolio_state.parquet
#   daily_pnl.parquet
#   metrics.json
#   config.yaml  # when config is attached
#   spec.yaml  # when config is attached
#   manifest.json

# Reload later
from ml4t.backtest.result import BacktestResult
result = BacktestResult.from_parquet("./results/my_backtest")
```

Engine results retain starting capital before the first observed NAV. A run starting
with $10,000 that pays a $10 fee before its first mark reports a first return of
-0.001, cumulative return of -0.001 and drawdown of -0.001. Observation timestamps
remain actual market observations. Older artifacts without a recorded initial
valuation retain their original first-observation baseline.

`manifest.json` identifies artifact schema version 3, or version 4 when the
result contains a prepared benchmark, and every component written.
Schema 3 records `initial_capital` separately from the first observed NAV. Schema 2
remains readable; its missing initial valuation is not reconstructed from configuration. Loading is
strict by default. An empty directory, missing manifest or required component, interrupted write,
malformed file, or unsupported schema version raises a specific `ArtifactError` subclass before a
result is returned. Selective exports are valid component exports, but they are not complete result
artifacts unless they contain all required components. Schema 1 was an unreleased development
format whose metrics used non-standard bare JSON constants for `NaN` and infinity; current readers
reject it rather than silently changing metric types. The stored `daily_pnl` component must decode
and equal the value recomputed from the stored equity curve.

Manifest-free beta artifacts require explicit recovery:

```python
result = BacktestResult.from_parquet("./results/beta_backtest", recovery=True)

for diagnostic in result.artifact_diagnostics:
    print(diagnostic.code, diagnostic.component, diagnostic.message)
```

Recovery reads supported components in a deterministic order and reports every missing, malformed,
or ignored component. Unsupported manifest schema versions still fail because their interpretation
is not defined. Missing `config` or `spec` data raises only when an explicit `include` requests it;
the default export records unavailable optional components in the manifest. A serialization failure
raises `ArtifactWriteError` whether the component came from the default set or an explicit
`include`. Component payloads are serialized before output files are created.
Non-finite metric floats use a tagged JSON object and are restored on load, so `metrics.json`
remains standards-compliant without losing `NaN` or infinity. NumPy arrays and Polars Series are
stored and loaded as JSON lists. The returned path mapping includes `manifest`; passing its keys
back through `include` treats that key as a no-op because the manifest is always written.

## Prepared benchmark returns

In a development build containing `BenchmarkData`, pass a prepared per-period
return table to `Engine`, `Engine.from_config` or `run_backtest`. Declare the
original source, financial return method and reinvestment convention. Backtest
retains the supplied decimal returns; it does not derive a dividend-adjusted
benchmark from raw execution prices.

This example uses a cash-dividend adjustment convention from the pinned
Zipline Reloaded 3.2.0 reference. The pre-ex-date close is 100 and the dividend
is 10, giving a retained previous-close factor of 0.9. The adjusted prior
close is therefore 90. Subsequent closes 95, 104.5 and 99 produce a cumulative
benchmark return of 10%. A raw-price-only series would report -1% from 100 to
99. A cash-dividend holding ledger is a separate account definition.

<!-- ml4t-doc-test: prepared-benchmark-returns -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, BenchmarkData, DataFeed, Engine, Strategy

times = [datetime(2025, 7, 1, 20, tzinfo=UTC), datetime(2025, 7, 2, 20, tzinfo=UTC),
         datetime(2025, 7, 3, 17, tzinfo=UTC)]
spec = {"time_contract_version": 2, "calendar": "NYSE", "data_frequency": "1d",
        "timestamp_semantics": "bar_close", "bar_label": "right"}
returns = pl.DataFrame({"timestamp": times,
                        "return": [95 / 90 - 1, 104.5 / 95 - 1, 99 / 104.5 - 1]})
benchmark = BenchmarkData(
    returns, source_id="fixture://declared-dividend-benchmark", asset="A",
    return_method="total_return", reinvestment="zipline_previous_close_cash_factor",
    feed_spec=spec,
)
prices = pl.DataFrame({"timestamp": times, "asset": ["A"] * 3,
                       "close": [95.0, 104.5, 99.0]})

class Cash(Strategy):
    def on_data(self, timestamp, data, context, broker):
        pass

result = Engine(DataFeed(prices_df=prices, feed_spec=spec), Cash(),
                BacktestConfig(initial_cash=1000), benchmark=benchmark).run()
retained = result.to_benchmark_dataframe()
cumulative = (retained["benchmark_return"] + 1).product() - 1
assert abs(cumulative - 0.1) < 1e-15
assert result.to_equity_dataframe()["equity"].to_list() == [1000.0] * 3
assert result.benchmark_metadata["return_method"] == "total_return"
print("Prepared benchmark periods and unchanged account verified.")
```

Timestamp values are timezone-aware physical period ends on the run's actual
observation grid. The constructor accepts Polars frames or a Pandas frame
with named timestamp columns. `return_col` selects a differently named return
column; its values must be finite decimal returns. Duplicate periods, missing
returns and missing observed periods raise errors. Backtest checks feed
coverage before strategy callbacks. It never forward-fills a daily percentage
across minute rows, converts session labels implicitly, or fills an absent
benchmark period with zero.

The actual Data preparation test in `tests/test_benchmark_data.py` passes the
reference's captured 0.9 factor through
`ml4t.data.adjustments.prepare_adjusted_history`, then consumes the resulting
return table through native Engine and the result artifact. It uses
[Data source c3b9db80](https://github.com/ml4t/data/commit/c3b9db80a66397420fb27adeccc762559dd41e07)
and Specs 0.2.0. This is a source-candidate integration. The report input uses
explicit retrospective supplied factors, with historical factor publication
left unknown; it is not a feature available to strategy decisions. Native Data
total-return preparation has its own declared method. Select the intended
method upstream and retain its provenance instead of relabelling it.

`to_parquet` writes `benchmark.parquet` and uses artifact schema 4 for results
with a benchmark. The manifest retains its source/method/reinvestment metadata
and a digest of the retained period keys and values. Strict reload validates
the digest, clock, coverage and provenance. Recovery reports an omitted or
invalid benchmark; it does not substitute one. Results without a benchmark
continue to write schema 3 and add no benchmark return columns. The current
reader accepts schemas 2, 3 and 4; older readers cannot read schema 4.

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

# Compute PortfolioAnalysis summary metrics
stats = analysis.compute_summary_stats()
print(f"Sharpe: {stats.sharpe_ratio:.2f}")
print(f"Max DD: {stats.max_drawdown:.2%}")
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
    Add `ml4t-diagnostic==0.1.4` for portfolio analysis. The full HTML
    tearsheet also requires the visualization extra:
    `uv add 'ml4t-diagnostic[viz]==0.1.4'`. Backtest imports and runs
    without this optional package.

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

Use the diagnostic integration function to build the HTML report from a
complete Backtest result. This example uses the bundled synthetic equity panel
and writes `tearsheet.html` in the current directory.

<!-- ml4t-doc-test: guide-tearsheet -->
```python
from pathlib import Path

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices
from ml4t.diagnostic.integration import generate_tearsheet_from_result

prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
result = Engine(
    DataFeed(prices_df=prices),
    ExampleRoundTrip("AAPL", 100),
    BacktestConfig(initial_cash=100_000),
).run()
html = generate_tearsheet_from_result(
    result, title="AAPL example", template="full", output_path="tearsheet.html"
)
assert "<html" in html.lower()
print(f"html_saved={Path('tearsheet.html').is_file()} fills={len(result.fills)}")
```

<!-- ml4t-doc-output: guide-tearsheet -->
```text
html_saved=True fills=2
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

## In the book

Chapter 16, Section 16.5, [Performance reporting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/09_performance_reporting.ipynb) develops return and drawdown interpretation. This page defines the result frames and artifact format used to reproduce those reports.

## Next Steps

- [Book Guide](../book-guide/index.md) -- notebook and case-study map for reporting workflows
- [Quickstart](../getting-started/quickstart.md) -- end-to-end examples
- [Profiles](profiles.md) -- compare results across framework profiles
