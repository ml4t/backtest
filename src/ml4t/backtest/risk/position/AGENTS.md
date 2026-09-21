# Position-risk guide

## Ownership

`static.py` contains fixed stop-loss and take-profit rules. `dynamic.py` contains rules whose state
changes with market data, `signal.py` handles signal exits, and `composite.py` combines rules behind
the protocol in `protocol.py`.

## Constraints

Evaluate rules against lifecycle-appropriate prices and preserve deterministic precedence when more
than one rule fires. Dynamic rule state is per position and must be cleared when that position
closes. Rule composition must retain the triggering rule and forced-exit cause for result reporting.
