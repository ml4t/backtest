# Analytics guide

## Ownership

This package derives metrics, equity curves, trade records, and diagnostic inputs from completed
backtest results. `metrics.py` owns performance statistics, `trades.py` owns round-trip analysis,
`equity.py` owns equity-series construction, and `bridge.py` is the optional `ml4t.diagnostic`
adapter.

## Constraints

Keep gross PnL, fees, slippage, and net PnL separately traceable. Partial closes must reconcile to
the underlying fills without duplicating entry costs. The diagnostic bridge must depend only on the
public Diagnostic API and must fail with an actionable optional-dependency error when it is absent.
