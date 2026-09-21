# Accounting guide

## Ownership

`policy.py` defines cash, margin, and crypto account policies. `gatekeeper.py` validates orders
against buying power before submission, and `account.py` holds account state.

## Constraints

Keep validation and mutation separate: a rejected order must not alter cash, margin, or reserved
buying power. Policy calculations must use the same notional, commission, and margin conventions as
the broker and fill path. Add behavioral coverage for long and short orders whenever those
conventions change.
