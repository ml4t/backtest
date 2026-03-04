# risk/position/ - 940 Lines

Position-level risk rules (stop-loss, trailing stop, take-profit).

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| dynamic.py | 505 | Trailing stop, dynamic stop-loss |
| static.py | 246 | Take-profit, fixed stop-loss |
| composite.py | 103 | Rule composition (RuleChain) |
| signal.py | 48 | Signal-based exit rules |
| protocol.py | 38 | Rule interface protocol |

## Key

`TrailingStop`, `StopLoss`, `TakeProfit`, `RuleChain`
