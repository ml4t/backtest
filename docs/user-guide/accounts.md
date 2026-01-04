# Account Policies

Configure cash or margin accounts for your backtests.

## Cash Account (Default)

No leverage - can only use available cash:

```python
from ml4t.backtest import BacktestConfig

config = BacktestConfig(
    initial_cash=100_000,
    margin_ratio=1.0,  # No leverage
)
```

## Margin Account

Use leverage for larger positions:

```python
config = BacktestConfig(
    initial_cash=100_000,
    margin_ratio=0.5,  # 2x leverage
)
```

Common margin ratios:

| Margin Ratio | Leverage | Description |
|--------------|----------|-------------|
| 1.0 | 1x | Cash account |
| 0.5 | 2x | Standard margin |
| 0.25 | 4x | Day trading margin |
| 0.1 | 10x | Futures-style margin |

## Transaction Costs

Configure realistic costs:

```python
config = BacktestConfig(
    initial_cash=100_000,
    commission=0.001,     # 0.1% per trade
    slippage=0.0005,      # 0.05% slippage
    margin_ratio=1.0,
)
```

## Presets

Use presets for common scenarios:

```python
from ml4t.backtest.presets import crypto_futures, us_equities

# Binance Futures (0.04% maker, 0.1x margin)
config = crypto_futures()

# US Equities via IB
config = us_equities()
```

## Insufficient Funds

Orders that exceed available buying power will:

1. Be rejected with `InsufficientFunds` error
2. Or be automatically sized down (if `auto_size=True`)

```python
config = BacktestConfig(
    initial_cash=100_000,
    auto_size=True,  # Automatically reduce order size
)
```
