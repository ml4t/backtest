# Broker core guide

## Ownership

These modules are broker collaborators, not a second public engine. `order_book.py` owns submission
and reservations, `execution_engine.py` orders eligible fills, `fill_engine.py` calculates fill
prices and quantities, and `risk_engine.py` evaluates position rules. `portfolio_ledger.py` records
portfolio state.

## Constraints

The broker remains the public facade and the owner of collaborator state. Do not introduce hidden
copies of cash, positions, orders, or prices. Preserve deterministic fill ordering and atomic
rejection: failed validation must leave every collaborator unchanged.
