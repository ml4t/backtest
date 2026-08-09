# Changelog

<!-- Generated from release/changelog.json. Do not edit managed entries. -->

## 0.1.0 - Unreleased

### Added

- Versioned compatibility snapshots, beta-artifact migration fixtures, commit-bound release
  candidates, and exact publication verification.
- Deterministic parity-claim generation and five reproducible performance workloads with behavior
  checksums.
- Installed-wheel execution checks for the primary installation and quickstart examples.
- A versioned causal strategy lifecycle and canonical pre-open target intents shared with
  ml4t-live, including child-order lineage, reconciliation evidence, and post-fill position-rule
  activation.

### Changed

- Broker state now has explicit account, market, order, risk, and execution-journal owners while
  the Broker remains the public facade.
- Intraday session enforcement now filters actual exchange intervals, including early closes and
  breaks, using the configured timezone for naive timestamps.
- DataFeed now rejects a missing resolved reference-price column at construction and documents
  close-only fallbacks explicitly.
- Stable distribution metadata, release-critical static analysis, and the 250-asset performance
  workload now match the 0.1 release contract.
- Weekly and month-end rebalance schedules require calendar metadata and use the final scheduled
  exchange session rather than the last available feed bar; holidays and missing period-end data
  no longer move a rebalance earlier.
- Online session schedules require evaluation on every event and expose validate_completed_run()
  to diagnose a missing final session close.
- Result artifacts retain intent and lifecycle counts by default. Set retain_intent_history or
  retain_lifecycle_history to include the corresponding event histories.

### Fixed

- Financial-accounting, order-rejection, artifact-integrity, quantity-tolerance, and
  parity-comparison defects found during the stable-release audit.
- Cross-framework validation now fails incomplete runs and requires exact retained comparisons for
  supported scenarios.

### Compatibility

- Stable support requires CPython 3.12 through 3.14 on Linux, macOS, and Windows; CPython 3.15
  prereleases are blocking compatibility checks until final release.
- Manifest-free 0.1.0b22 result artifacts require explicit recovery mode and can then be rewritten
  as schema-versioned stable artifacts.
- The ambiguous beta-only Broker(account_type=...) constructor spelling was removed; use explicit
  short-selling and leverage policy fields.
- Engine instances are single-use in 0.1.0; create a new Engine for each independent run.
- BacktestResult fills, equity_curve, and portfolio_state lists are snapshots instead of aliases
  to mutable Engine or Broker state.
- The beta-only on_before_risk callback and future-timestamp on_prepare signature are removed.
  Engine construction raises HistoricalStrategyCompatibilityError for strategies that require
  either historical behavior.
- TargetWeightExecutor.prepare_schedule was removed. Configure calendar, timezone,
  session_start_time, data_frequency, and timestamp_semantics on RebalanceConfig for causal
  schedule evaluation.
- ExecutionPolicy.liquidity_fraction now applies only to opening-auction target children.
  Configure Engine execution_limits explicitly to constrain ordinary strategy orders.

### Validation

- Release requires compatibility, correctness, coverage, documentation, lint, packaging, parity,
  performance, security, and type-check gates bound to one commit.
- The exact wheel and source distribution that pass those gates are retained as the release
  candidate and verified after publication.

## 0.1.0b22 - 2026-07-25

### Added

- Strategy.on_before_risk(), explicit position-rule clearing, and an opt-in entry-bar
  trailing-watermark mode.

### Compatibility

- Strategies that do not override on_before_risk retain their prior behavior. The callback remains
  beta-only and is scheduled for explicit migration rather than inclusion in the stable lifecycle.

## 0.1.0b21 - 2026-06-12

### Added

- Broker.equity() as the preferred marked-account-equity accessor.

### Compatibility

- Broker.get_account_value() remains available and delegates to equity().

## 0.1.0b20 - 2026-05-17

### Fixed

- RiskManager.update(..., broker=...) now applies liquidation actions by flattening positions
  instead of only reporting the breach.

### Compatibility

- Liquidation behavior changes only when a portfolio limit returns action='liquidate' and a broker
  is supplied. Omitting the broker retains reporting behavior and emits a warning.

## 0.1.0b19 - 2026-05-17

### Added

- Broker.flatten_all_positions() and the typed risk_liquidation exit reason for portfolio-wide
  breaches.

### Compatibility

- Existing strategy exits are unchanged; portfolio risk integrations can opt into explicit
  flattening and preserve the liquidation cause.

## 0.1.0b18 - 2026-05-16

### Added

- Per-symbol percentage-of-notional margin schedules for price-aware futures margin assumptions.

### Compatibility

- Existing global and fixed-dollar margin settings remain valid. Defining the same asset in
  fixed_margin_schedule and margin_pct_schedule is rejected as ambiguous.

## 0.1.0b17 - 2026-05-10

### Changed

- `DataFeed` now indexes each timestamp as an `(offset, length)` slice over the original
  prices, signals, and context frames instead of eagerly materializing one child
  `DataFrame` per timestamp.
- This substantially reduces timestamp-partition memory overhead on large multi-asset
  panels while preserving the same lazy per-bar iteration contract.
- Added regression coverage for unsorted input alignment and updated the dedicated
  DataFeed memory tests to validate the new slice-index storage model.

### Validation

- `uv run pytest tests/ -q`
- `uv run ruff check src/ml4t/backtest tests`
- `uv run ty check`
- `uv run python -m mkdocs build --strict`

## 0.1.0b16 - 2026-05-05

### Added

- `SlippageType.SPREAD` with `SpreadSlippage` for bar-based spread-cost modeling.
- Asset-aware spread configuration via `slippage_spread`, `slippage_spread_by_asset`, and
  `slippage_spread_convention`.
- Explicit broker assumption helpers via `BacktestConfig.from_assumptions(...)` and
  `BacktestConfig.from_user_config(...)`.
- The `ibkr_us_stocks_fixed` preset for explicit Interactive Brokers US equities fixed
  commission assumptions.

### Changed

- Generic engine defaults are now neutral again: no default commission, no default slippage,
  `min_trade_value=0.0`, and `min_weight_change=0.0`.
- Integer shares are now the default share mode for the generic and fast profiles, while
  `vectorbt` remains fractional.
- Integer rebalancing now rounds target-share deltas to the nearest whole share instead of
  truncating toward zero, reducing correction churn and fill amplification.
- User guides now document neutral defaults, broker assumptions, spread slippage, and the
  updated rebalance behavior.

### Validation

- `uv run pytest tests/ -q`
- `uv run ruff check src/ml4t/backtest tests`
- `uv run ty check`
- `uv run python -m mkdocs build --strict`

## 0.1.0b15 - 2026-04-27

### Added

- `OrderType.MOC` as a first-class market-on-close order primitive.
- `Broker.close_position(..., order_type=...)` so positions can be flattened with
  `OrderType.MOC` directly.

### Changed

- `MOC` orders now execute at the current bar's close instead of following the standard
  `NEXT_BAR` market-order timing.
- User docs now describe `MOC` order timing, daily-bar semantics, and position-closing
  usage.

### Validation

- `uv run pytest tests/test_order_type_matrix.py tests/test_order_type_moc.py -q`
- `uv run ruff check src/ml4t/backtest/broker.py src/ml4t/backtest/core/execution_engine.py src/ml4t/backtest/core/fill_engine.py src/ml4t/backtest/core/order_book.py src/ml4t/backtest/engine.py src/ml4t/backtest/types.py tests/test_order_type_matrix.py tests/test_order_type_moc.py`
- `uv run python -m mkdocs build --strict`

## 0.1.0b13 - 2026-04-01

### Changed

- `ml4t-backtest` now depends on `ml4t-specs` for shared market-data artifact contracts
  instead of importing `FeedSpec`, `MarketDataSpec`, and serialization helpers from
  `ml4t-data`.
- Runtime config, data-feed, result, annualization, and rebalance-schedule code now import
  shared contract types from `ml4t.specs`.
- `spec_bridge.market_data_spec_to_feed_spec()` now delegates to `FeedSpec.from_any()` so
  spec-to-runtime projection stays aligned with the canonical contract package.
- Contract, feed, broker, result, rebalancer, schedule, and artifact-spec tests now validate
  the `ml4t-specs` path directly.
- User docs now show `FeedSpec` imports from `ml4t.specs`.

### Validation

- `uv run pytest tests/ -q`
- `uv run ty check`
- `uv run python -m mkdocs build --strict`

## 0.1.0b12 - 2026-03-29

### Added

- Internal shipped validation helpers under `ml4t.backtest._validation`, including:
  - `lean_runner.py` for shared LEAN CLI orchestration, data export, and artifact parsing
  - `vectorbt_runner.py` for shared VectorBT matrix prep, execution, and result extraction
  - `backtrader_runner.py` for shared Backtrader target-share execution and PyFolio parsing
  - `zipline_runner.py` for shared Zipline bundle orchestration and transaction parsing

### Changed

- The library validation harness now uses the shared `_validation` helpers instead of
  carrying duplicate LEAN, VectorBT, Backtrader, and Zipline integration logic inside
  `validation/benchmark_suite.py`.
- Chapter 16 book parity code can now rely on the released `ml4t-backtest` package for
  shared cross-engine validation machinery instead of vendoring those heavy helpers.

### Validation

- `uv run pytest tests/benchmark/test_lean_adapter.py tests/benchmark/test_backtrader_zipline_runners.py tests/test_config_wiring.py -q`
- `uv run python -m py_compile src/ml4t/backtest/_validation/lean_runner.py src/ml4t/backtest/_validation/vectorbt_runner.py src/ml4t/backtest/_validation/backtrader_runner.py src/ml4t/backtest/_validation/zipline_runner.py validation/benchmark_suite.py`

## 0.1.0b11 - 2026-03-24

### Added

- `BacktestResult.predictions` and `BacktestResult.to_predictions_dataframe()` to preserve
  the raw prediction or model-input DataFrame passed into the backtest.
- `predictions.parquet` export/import support in `BacktestResult.to_parquet()` and
  `BacktestResult.from_parquet()`.

### Changed

- Engine results now treat the raw `signals_df` input surface as predictions for downstream
  diagnostics, matching `ml4t-diagnostic`'s current contract.
- Parquet import falls back from legacy `signals.parquet` to the new predictions surface.
- User guides and README now document the raw predictions surface for downstream analysis.

### Validation

- `uv run ruff check src/ml4t/backtest/result.py src/ml4t/backtest/engine.py tests/test_result.py tests/test_core.py`
- `uv run pytest tests/test_result.py tests/test_core.py -q`
- `uv run ty check`
- `uv run python -m mkdocs build --strict`

## 0.1.0b10 - 2026-03-24

### Added

- `BacktestConfig` support for serialized top-level `feed` and passthrough `metadata`
  sections, enabling sparse input presets with generic provenance fields.
- `BacktestResult.to_spec_dict()` for a resolved runtime snapshot containing the full
  replayable config, library version, and realized run window.
- `spec.yaml` export alongside `config.yaml` in `BacktestResult.to_parquet()`.

### Changed

- `BacktestConfig.to_dict()` now emits plain-data feed metadata that round-trips safely
  through dict, YAML, and Parquet persistence workflows.
- `BacktestResult.from_parquet()` now falls back to `spec.yaml` when `config.yaml` is absent.
- User guides now document the config workflow for sparse input, resolved output, `feed`,
  `metadata`, and reproducibility exports.

### Validation

- `uv run ruff check src/ml4t/backtest/config.py src/ml4t/backtest/result.py tests/test_broker.py tests/test_result.py`
- `uv run pytest tests/test_broker.py tests/test_result.py -q`
- `uv run ty check`
- `uv run python -m mkdocs build --strict`

## 0.1.0b9 - 2026-03-24

### Added

- Quote-aware `DataFeed` support for `price_col`, bid, ask, midpoint, and quote-size caches.
- New `ExecutionPrice` sources: `price`, `bid`, `ask`, `quote_mid`, and `quote_side`.
- Separate `mark_price` configuration for open-position marking.
- `BacktestResult.to_fills_dataframe()` and persisted `fills.parquet` export/import.
- `BacktestResult.to_portfolio_state_dataframe()` and persisted `portfolio_state.parquet`.
- Quote context fields on `Fill` and summarized quote context on `Trade`.
- Activity and exposure metrics: `num_fills`, `num_rebalance_events`, `unique_symbols_traded`,
  `total_filled_notional`, `avg_turnover`, `max_turnover`, `avg_open_positions`,
  and `max_open_positions`.

### Changed

- `FeedSpec.price_col` now drives the broker reference price instead of being collapsed back to `close`.
- Market execution can use side-aware quotes: buys at ask, sells at bid.
- `QUOTE_SIDE` marking prices long inventory on the bid and short inventory on the ask.
- Result persistence now includes fills alongside trades, equity, daily P&L, metrics, and config.
- Result persistence now includes portfolio state alongside trades, fills, equity,
  daily P&L, metrics, and config.
- User guides and README now document quote-aware feeds, mark pricing, and fill export.
- User guides and README now document portfolio-state reporting and quote-aware audit fields.

### Performance

- Legacy OHLCV hot path remains faster than the pre-optimization baseline.
- Quote-aware execution adds moderate overhead relative to the optimized OHLCV path, while staying ahead of the legacy baseline in local benchmarks.

### Validation

- `uv run ty check`
- `pre-commit run --all-files`
- `uv run pytest tests/ -q`
