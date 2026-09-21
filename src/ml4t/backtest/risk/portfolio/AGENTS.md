# Portfolio-risk guide

## Ownership

`limits.py` implements aggregate exposure, drawdown, and position-count constraints. `manager.py`
evaluates configured rules and returns the resulting action to the broker.

## Constraints

Keep limit evaluation side-effect free. State changes belong to the broker after it accepts an
explicit risk action. Define whether a breach rejects a new order, halts new risk, or liquidates
positions, and preserve that reason in results. Cover boundary values and recovery behavior.
