# ml4t.backtest package guide

## Ownership

This package contains the public engine, broker, result, configuration, data-feed, and strategy
interfaces. Preserve the exports in `__init__.py`; implementation modules may change without moving
user code away from those package-level imports.

The engine owns event ordering. The broker owns orders, fills, positions, cash, and collaborator
state. Configuration profiles make execution assumptions explicit. Results and exported artifacts
are compatibility surfaces, so schema changes require migration coverage.

## Constraints

Use the nearest subpackage guide for accounting, analytics, broker internals, execution, and risk.
Keep lifecycle and order protocols runtime-neutral in `ml4t.specs`; adapt them here instead of
creating a second protocol.
