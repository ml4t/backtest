# core/ - 1,365 Lines

Decomposed broker internals. Extracted from broker.py during refactoring.

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| order_book.py | 517 | Order submission, shadow cash, immediate fill |
| execution_engine.py | 407 | Fill ordering (EXIT_FIRST, FIFO, SEQUENTIAL) |
| fill_engine.py | 222 | Fill price calculation, share rounding |
| risk_engine.py | 157 | Position rule evaluation, deferred exits |
| portfolio_ledger.py | 31 | Ledger tracking |
| shared.py | 31 | Shared types (SubmitOrderOptions) |

## Key

`OrderBook`, `ExecutionEngine`, `FillEngine`, `RiskEngine`
