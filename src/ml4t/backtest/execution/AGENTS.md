# Execution guide

## Ownership

`fill_executor.py` simulates fills and costs. `rebalancer.py` converts target weights into orders,
while `impact.py` and `limits.py` define explicit market-impact and order-limit policies.

## Constraints

Execution must be causal and deterministic for a fixed event stream and configuration. Apply share
rounding, volume participation, limits, slippage, and commission in a documented order, and retain
the reason for any rejected or constrained order. Rebalancing must handle missing prices without
creating invalid quantities or mutating unaffected positions.
