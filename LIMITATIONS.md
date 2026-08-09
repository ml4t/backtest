# ml4t-backtest Known Limitations and Assumptions

This document defines what the simulator does not infer or model. A backtest is
only as reliable as its input data and configured execution assumptions.

## Data and Time

- The default feed requires a timestamp, a recognized entity column, and `close`.
  A custom `FeedSpec` may map those roles to other columns.
- Open, high, low, and volume are optional. Missing OHLC values fall back to the
  configured close and missing volume becomes zero. Close-only data cannot support
  reliable stop, limit, or volume-limited execution assumptions.
- The feed iterates the sorted union of price, signal, and context timestamps.
  Signal-only or context-only timestamps produce an empty per-asset data mapping.
- Input prices must already reflect the corporate-action treatment required by the
  strategy. The library does not adjust splits, dividends, symbol changes, mergers,
  bankruptcies, or delistings.
- Naive timestamps use the configured feed timezone. With `enforce_sessions=True`,
  daily bars are filtered to valid trading dates and intraday bars are filtered to
  exchange session intervals, including early closes and configured breaks.

## Temporal and Execution Model

- The simulator is bar-based. OHLC values do not identify the order in which prices
  occurred within a bar.
- `SAME_BAR` lets a strategy observe a bar and fill at that bar's close. This can
  introduce look-ahead bias unless the input signal was available before the fill.
- `OrderType.MOC` submitted by `on_data` fills at the current close even in
  `NEXT_BAR` mode. Users must ensure the decision was available before the auction.
- Limit and stop fills use configured bar-price rules. The simulator does not model
  exchange queue position or reconstruct order-book events from OHLCV.
- `ExecutionLimits` can constrain fills from bar volume or quote size. Without an
  execution limit, the configured price is assumed available for the requested size.
- Market impact models are explicit approximations. They do not model venue routing,
  latency, hidden liquidity, spread dynamics, order acknowledgements, cancellation
  delays, exchange halts, or auction imbalances.

## Accounts and Cash Flows

- `settlement_delay` is measured in processed bars, not calendar or business days.
  It does not model partial settlement, settlement failure, or clearing fees.
- Margin requirements use configured account-wide rates or per-asset fixed or
  percentage schedules. They are not a historical broker risk engine.
- Portfolio risk breaches and maintenance-margin failures can trigger liquidation,
  but the resulting bar-based fills remain subject to configured execution rules.
- The account is single-currency. Foreign exchange conversion, interest, dividends,
  futures variation margin, perpetual funding, short borrow availability and fees,
  taxes, and wash-sale accounting are not modeled.
- Regulatory and broker-specific rules such as locate requirements, price tests,
  day-trading limits, and account permissions are outside the generic account model.

## Numerical Representation

- Prices, quantities, and cash use double-precision floating-point values. Integer
  share mode rounds quantities but does not convert accounting to decimal arithmetic.
- Contract multipliers and margin schedules must be supplied for instruments whose
  economics differ from one currency unit per quantity unit.

## Strategy Lifecycle Compatibility

- Lifecycle V1 is shared with `ml4t-live`. A strategy that defines the removed
  `on_before_risk` callback or requests future timestamps through the historical
  `on_prepare` signature is rejected during engine construction.
- Opening-auction targets require a canonical intent whose decision time and
  information cutoff precede the auction. A target registered after its pre-open
  phase is rejected instead of receiving a historical opening fill.
- Daily OHLC data does not reveal whether the high or low occurred first. A position
  rule activated after an opening fill therefore follows the configured bar-path
  policy and can reject an ambiguous outcome.

## Validation Boundaries

- Framework profiles reproduce the retained validation scenarios and cohorts. This
  evidence does not prove equivalence for untested framework versions, data shapes,
  custom extensions, or parameter combinations.
- Parity claims in README and the profiles guide are generated from retained evidence.
  A missing or unsupported scenario is not counted as a pass.
- Runtime and memory evidence depends on interpreter, hardware, operating system, and
  workload. The project retains reproducible workloads and behavior checksums but does
  not publish hardware-dependent performance values as stable guarantees.
