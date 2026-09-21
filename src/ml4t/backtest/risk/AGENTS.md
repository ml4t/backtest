# Risk guide

## Ownership

Position rules decide when an individual holding should exit. Portfolio rules constrain aggregate
exposure, drawdown, and position counts. Shared result and configuration types belong in
`types.py`; implementations belong in the nearest subpackage.

## Constraints

Risk evaluation may inspect only state available at its lifecycle phase. Forced exits must retain
their cause through order creation and fill reporting. A portfolio halt must have explicit resume or
liquidation semantics rather than silently leaving positions unmanaged.
