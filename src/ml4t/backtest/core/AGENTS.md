# core/ - 1,314 Lines

Decomposed broker internals. Extracted from broker.py during refactoring.

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| order_book.py | 500 | Order submission, shadow cash, immediate fill |
| execution_engine.py | 348 | Fill ordering (EXIT_FIRST, FIFO, SEQUENTIAL) |
| fill_engine.py | 222 | Fill price calculation, share rounding |
| risk_engine.py | 157 | Position rule evaluation, deferred exits |
| shared.py | 54 | Shared types (SubmitOrderOptions) |
| portfolio_ledger.py | 33 | Ledger tracking |

## Key

`OrderBook`, `ExecutionEngine`, `FillEngine`, `RiskEngine`
