# Reconcile trading costs and funding

The [first backtest](../getting-started/quickstart.md) charged no trading costs.
These examples keep the same synthetic inputs fixed while adding commission,
slippage, market impact, and perpetual funding. Each printed amount is
recomputed by the installed-wheel documentation check.

## Compare gross and net equity

The AAPL strategy buys 1,000 shares and closes after its fifth bar. All three
runs use the same orders and price panel. The `gross` run has no costs. The
`moderate` assumptions leave a positive $2,392.44 change in equity; the `high`
assumptions turn the same $3,000 before-cost trade into a $2,771.76 loss.
These labels describe only the declared synthetic scenarios.

<!-- ml4t-doc-test: tutorial-costs-equity -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, CommissionType, DataFeed, Engine
from ml4t.backtest.config import SlippageType
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices
from ml4t.backtest.execution import LinearImpact


prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
scenarios = [
    ("gross", 0.0, 0.0, 0.0),
    ("moderate", 0.001, 0.0005, 0.1),
    ("high", 0.01, 0.005, 0.2),
]
print("ml4t-backtest " + version("ml4t-backtest"))
baseline_fills = None
baseline_gross = None
for name, fee_rate, slip_rate, impact in scenarios:
    config = BacktestConfig(
        initial_cash=1_000_000,
        commission_type=CommissionType.PERCENTAGE,
        commission_rate=fee_rate,
        slippage_type=SlippageType.PERCENTAGE,
        slippage_rate=slip_rate,
    )
    result = Engine(
        DataFeed(prices_df=prices), ExampleRoundTrip("AAPL", 1000), config,
        market_impact_model=LinearImpact(coefficient=impact),
    ).run()
    trade = result.trades[0]
    if baseline_fills is None:
        baseline_fills = result.fills
        baseline_gross = trade.gross_pnl
    assert baseline_gross is not None
    fees = sum(fill.commission for fill in result.fills)
    slip = trade.total_slippage_cost
    price_drag = sum(
        (fill.price - baseline.price) * fill.quantity
        if fill.side.value == "buy" else (baseline.price - fill.price) * fill.quantity
        for baseline, fill in zip(baseline_fills, result.fills, strict=True)
    )
    impact_cost = price_drag - slip
    net = result.metrics["final_value"] - 1_000_000
    assert abs(baseline_gross - fees - slip - impact_cost - net) < 0.01
    print(f"{name}: gross={baseline_gross:.2f} fees={fees:.2f} "
          f"slip={slip:.2f} impact={impact_cost:.2f} net={net:.2f}")
```

<!-- ml4t-doc-output: tutorial-costs-equity -->
```text
ml4t-backtest {package_version}
gross: gross=3000.00 fees=0.00 slip=0.00 impact=0.00 net=3000.00
moderate: gross=3000.00 fees=379.73 slip=189.87 impact=37.96 net=2392.44
high: gross=3000.00 fees=3797.18 slip=1898.66 impact=75.92 net=-2771.76
```

The `gross` column is the no-cost run's completed trade P&L. Commission comes
from the fill records. The trade's `total_slippage_cost` reports the configured
slippage model. The script infers impact as the remaining change in execution
prices relative to the no-cost fills. For each row, gross minus commission,
slippage, and impact equals the change from the $1,000,000 starting equity.
`trade.gross_pnl` in a costed run already reflects its actual fill prices, so
subtracting slippage from that field again would double count it. See
[Market Impact](../user-guide/market-impact.md) for the individual models.

## Add perpetual funding

A long BTC perpetual receives two funding events while it is held. A positive
rate debits the position and a negative rate credits it. The event is processed
before orders at the same timestamp; the January 7 payment uses the position
held just before its exit. The example uses a synthetic price panel and declared
funding rates, with no trading fees.

<!-- ml4t-doc-test: tutorial-costs-funding -->
```python
from datetime import datetime
from importlib.metadata import version

import polars as pl
from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices


prices = load_example_prices("crypto_perp").filter(pl.col("asset") == "BTC-PERP")
funding = pl.DataFrame({
    "timestamp": [datetime(2024, 1, 4), datetime(2024, 1, 7)],
    "asset": ["BTC-PERP", "BTC-PERP"],
    "rate": [0.001, -0.0005],
})
config = BacktestConfig(initial_cash=100_000, allow_leverage=True)
spec = ContractSpec("BTC-PERP", AssetClass.FUTURE, margin=4000)
base = Engine(DataFeed(prices_df=prices), ExampleRoundTrip("BTC-PERP", 1),
              config, contract_specs={"BTC-PERP": spec}).run()
funded = Engine(DataFeed(prices_df=prices), ExampleRoundTrip("BTC-PERP", 1),
                config, contract_specs={"BTC-PERP": spec}, funding_df=funding).run()
print("ml4t-backtest " + version("ml4t-backtest"))
for row in funded.to_funding_dataframe().iter_rows(named=True):
    print(f"{row['timestamp']:%Y-%m-%d} rate={row['rate']:.4f} cash={row['cash_delta']:.2f}")
print(f"trading P&L: ${base.trades[0].pnl:.2f}")
print(f"funding: -${abs(funded.metrics['total_funding']):.2f}")
print(f"net change: ${funded.metrics['final_value'] - 100_000:.2f}")
assert abs(base.trades[0].pnl + funded.metrics['total_funding']
           - (funded.metrics['final_value'] - 100_000)) < 0.01
```

<!-- ml4t-doc-output: tutorial-costs-funding -->
```text
ml4t-backtest {package_version}
2024-01-04 rate=0.0010 cash=-42.20
2024-01-07 rate=-0.0005 cash=21.75
trading P&L: $700.00
funding: -$20.45
net change: $679.55
```

`to_funding_dataframe()` keeps funding separate from fills and closed trades.
The $700 trading gain minus $20.45 funding equals the $679.55 change in equity.
The rate is applied to the held quantity, current causal mark, and contract
multiplier; see [Funding payments](../user-guide/market-impact.md#perpetual-futures-funding)
for the complete input and event-time contract.

## Charge a flip once

A position flip closes a long position and opens a short position with one
sell order. With a $5 per-trade commission, the 20-share flip creates one fill
and one $5 charge. The earlier buy is a separate $5 order.

<!-- ml4t-doc-test: tutorial-costs-flip -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, CommissionType, DataFeed, Engine, OrderSide, Strategy
from ml4t.backtest.example_data import load_example_prices


class FlipOnce(Strategy):
    def __init__(self):
        self.asset_bars = 0

    def on_data(self, timestamp, data, context, broker):
        self.asset_bars += 1
        if self.asset_bars == 1:
            broker.submit_order("AAPL", 10, OrderSide.BUY)
        elif self.asset_bars == 3:
            broker.submit_order("AAPL", 20, OrderSide.SELL)


prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
result = Engine(
    DataFeed(prices_df=prices), FlipOnce(),
    BacktestConfig(
        initial_cash=100_000, allow_short_selling=True,
        commission_type=CommissionType.PER_TRADE, commission_per_trade=5,
    ),
).run()
print("ml4t-backtest " + version("ml4t-backtest"))
for fill in result.fills:
    print(f"{fill.timestamp:%Y-%m-%d} {fill.side.value} {fill.quantity:g} commission=${fill.commission:.2f}")
print(f"total commission: ${result.metrics['total_commission']:.2f}")
```

<!-- ml4t-doc-output: tutorial-costs-flip -->
```text
ml4t-backtest {package_version}
2024-01-03 buy 10 commission=$5.00
2024-01-05 sell 20 commission=$5.00
total commission: $10.00
```

## In the book

Chapter 18, Section 18.7, [Transaction cost analysis and model validation](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/18_transaction_costs/README.md),
and [notebook 10, Gross versus net performance](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/18_transaction_costs/10_gross_vs_net_performance.ipynb)
extend the cost decomposition to larger strategy runs. The [crypto-perpetual
cost notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/case_studies/crypto_perps_funding/16_costs.ipynb)
adds case-study funding and fee assumptions.
