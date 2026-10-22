# Changelog

<!-- Generated from release/changelog.json. Do not edit managed entries. -->

## 0.2.0 - Unreleased

### Added

- Run causal current/history, scheduled callbacks, prepared Pipeline and frozen CSV inputs through
  native runtime services and the optional Zipline callback facade. Original OLMAR, moving-average
  and momentum journeys retain declared timing and execution methods.
- Apply raw splits, cash and stock distributions, mergers and capital flows to native positions,
  pending orders and account histories. Retain action, cash-flow and dated-roll ledgers in
  versioned result artifacts.
- Offer explicit futures variation-cash accounting, declared settlement policies, dated-contract
  roll legs and lifecycle controls. Continuous-future research reads use prepared Data inputs;
  orders execute dated contracts at raw quotes.
- Consume immutable Data snapshots and captured Engineer feature declarations, retaining exact
  input provenance through native results. Add runnable offline preparation-to-Diagnostic and
  continuous-futures migration workflows.

### Fixed

- Compute first-period investment returns from configured initial capital, retain observed
  account/holdings/fill histories, and distinguish external capital, futures exposure, liquidation
  value and NAV.
- Reject duplicate or ambiguous event keys, changed verified inputs, and undeclared
  continuous-future queries before simulation callbacks or orders.
- Qualify actual producer workflows on current Pandas, allowing only the exact deprecated weekday
  alias from upstream PMC NYSE schedule construction while keeping other warnings fatal.

### Compatibility

- Calendar selection now distinguishes omission, a named calendar and explicit None: inherit
  declared feed metadata then NYSE only when omitted; None disables schedules. Unversioned saved
  configurations retain their legacy interpretation. Calendar-free numerical annualization
  defaults to 252 observations per year and accepts an explicit periods_per_year.
- Require ml4t-specs>=0.2.0. Preserve historical result readers and the v0.1 compatibility
  snapshot while recording the expanded v0.2 public surface. Native cash accounting remains the
  default; variation accounting and legacy execution conventions are opt-in.
- The complete optional Data/Engineer/Diagnostic migration cohort is qualified on Python 3.12
  through 3.14 against documented immutable source builds. Backtest core Python 3.15 qualification
  is separate; these source identities are not claims about older published consumer wheels.

### Validation

- Expand installed successor qualification to actual causal history, Pipeline, identity,
  lifecycle, controls and account producer paths across Linux, macOS and Windows. Require every
  selected workflow to execute without skips. The Arrow-present cohort uses PyArrow 26 or later
  with Pandas 3.1 to avoid deprecated timezone-dropping index access in older Arrow conversion.
- Retain accepted performance measurements within their explicitly reviewed scope and unchanged
  output identities; no new speed claim or reference timing measurement is introduced.

## 0.1.13 - 2026-10-15

### Fixed

- Retain Sunday evening bars that belong to a Monday CME exchange session.
- Apply max_single_weight symmetrically to requested long and short target weights.
- Include futures contract multipliers in the affected simple-cash and sequential buying-power
  reservation checks.
- Return positive infinite CAGR when annualizing an extreme short positive-return window
  overflows, so EquityCurve.to_dict completes.
- Keep the shared comparison group compatible with Plotly below 7 on Python below 3.15 and
  regenerate all isolated lock metadata.

### Documentation

- Correct executor-mode configuration, fill-eligibility descriptions, headroom and whole-share
  target rounding. Link the rebalancing guide to the enum, configuration and executor API
  reference.
- Make the Zipline migration guide explicit about input and execution assumptions, with a checked
  synthetic target-weight example and a link to the tested Diagnostic portfolio-analysis handoff.
- Publish the local contribution guide and issue/PR templates linked to the shared policy,
  preserving private security reporting and the verified usage-documentation destination.

### Build and compatibility

- Update the build backend, pinned workflow actions and vulnerable dependency locks while
  preserving the frozen comparison framework identities.
- Accept final CPython 3.15 in the canary and describe its steps accurately. Stable support
  remains CPython 3.12 through 3.14; no blanket optional-package support for 3.15 is added.

### Compatibility

- Results change in the corrected overnight-session, short-target and futures-cash cases. Target
  caps apply before whole unit rounding and do not guarantee a continuous limit on realized
  exposure.
- Consumers of raw metric dictionaries must handle non-finite CAGR for extreme short windows.
  Versioned result exports retain their documented tagged JSON encoding for non-finite metrics.

### Validation

- Regenerate correctness evidence from frozen inputs and preserve original timing measurements,
  using the existing source-certification mechanism for documentation-only source changes.

## 0.1.12 - 2026-09-24

### Changed

- Complete task guides and the verified Book Guide with installed-wheel examples.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.12).

## 0.1.11 - 2026-09-24

### Changed

- Correct User Guide API examples and explanations of strategy state.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.11).

## 0.1.10 - 2026-09-24

### Changed

- Add an audited User Guide with task navigation.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.10).

## 0.1.9 - 2026-09-24

### Changed

- Correct backtest behavior, add guided documentation and refresh the build backend.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.9).

## 0.1.8 - 2026-09-21

### Changed

- Reconcile time zones before trade enrichment, improve verification of newly published packages,
  and complete the closure documentation audit.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.8).

## 0.1.7 - 2026-09-12

### Changed

- Remove the inert LinearImpact.permanent_fraction parameter and add provenance-preserving source
  certification for performance evidence.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.7).

## 0.1.6 - 2026-09-09

### Changed

- Standardize ecosystem setup and public agent orientation.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.6).

## 0.1.5 - 2026-09-05

### Changed

- Publish bounded real-strategy comparisons and timing evidence, and authenticate licensed
  VectorBT Pro installation through GitHub CLI.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.5).

## 0.1.4 - 2026-09-03

### Changed

- Use the feed VWAP for VWAP execution and reject unavailable required VWAP prices.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.4).

## 0.1.3 - 2026-08-12

### Changed

- Update pinned GitHub Actions.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.3).

## 0.1.2 - 2026-08-11 (tag only)

### Tagged source

- The annotated v0.1.2 tag exists. No matching published release or artifact is currently listed
  in the GitHub/PyPI records checked on 2026-10-15. See [tagged
  source](https://github.com/ml4t/backtest/tree/v0.1.2).

## 0.1.1 - 2026-08-10

### Changed

- Add per-asset position rules for pre-open execution.

### Release notes

- [Published release notes](https://github.com/ml4t/backtest/releases/tag/v0.1.1).

## 0.1.0 - 2026-08-09

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
  exchange session rather than the last available feed bar. Holidays do not move a rebalance
  earlier, and a missing completed period-end session raises a feed-completeness error.
- Explicit rebalance timestamps match absolute instants across timezone representations and online
  evaluation requires chronological events. A scheduled instant inside the observed feed window
  that has no matching event raises an alignment error instead of being skipped silently.
- Batch and online session schedules require every complete exchange session to match its required
  close; only an incomplete final session is exempt, and calendar mismatches fail explicitly.
- Result artifacts retain intent and lifecycle counts by default. Set retain_intent_history or
  retain_lifecycle_history to include the corresponding event histories.
- Morning-start exchange sessions group pre-market and regular-hours events under the same local
  calendar date; exchange session labels and opens come from cached authoritative calendar
  schedules.
- Scheduled TargetWeightExecutor.execute calls register completion validation with the Engine, so
  missing required closes and partial event coverage fail the run with distinct diagnostics.
- Deferred next-open risk exits fill before opening targets are sized. Fractional accounts reject
  largest-remainder residual handling regardless of rounding because its whole-unit allocation
  cannot redistribute fractional remainders; use keep-cash instead.

### Fixed

- Financial-accounting, order-rejection, artifact-integrity, quantity-tolerance, and
  parity-comparison defects found during the stable-release audit.
- Cross-framework validation now fails incomplete runs and requires exact retained comparisons for
  supported scenarios.
- Batch and online session schedules apply the same daily-close fallback when data frequency and
  timestamp semantics are both omitted. Batch evaluation and flagless online evaluation reject
  repeated dates and intervals under 12 hours as ambiguous intraday data; online events may
  instead carry is_session_close consistently, but mixing explicit and inferred boundaries on one
  date raises.
- Session-aligned P&L and broker statistics interpret naive timestamps in the configured data
  timezone before applying the authoritative exchange timezone.

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
- Custom morning session_start_time values are rejected; morning-start sessions always use the
  local calendar date. Omit the override or restate the configured exchange calendar's standard
  open.
- Evening session_start_time overrides remain available for calendar-free and evening-start
  sessions, but are rejected for calendars whose authoritative open is in the morning.
- SessionConfig is immutable; construct and install a replacement instead of mutating calendar,
  timezone, or session_start_time after configuration.
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
