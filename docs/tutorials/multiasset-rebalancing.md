# Turn signals into portfolio weights

The [first backtest](../getting-started/quickstart.md) held one stock. This
tutorial uses the same [bundled synthetic panels](data.md) to rebalance two
assets. A `signals_df` row contains one declared `prediction` for each
`(timestamp, asset)` pair. The values 0.7 and 0.3 are fixed teaching inputs,
not fitted forecasts. The strategy puts 80% of capital to work, so its first
targets are 56% and 24%, with 20% left in cash. It evaluates its schedule on
every callback and rebalances every second session.

## Equities and ETFs

Both assets in each of these panels share the same daily bar timestamp. The
NYSE calendar and New York timezone identify the synthetic 16:00 local closes.
Orders submitted at the January 2 decision fill on January 3.

<!-- ml4t-doc-test: tutorial-multiasset-equity-etf -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, RebalanceConfig, RebalanceSchedule, Strategy, TargetWeightExecutor
from ml4t.backtest.example_data import load_example_prices


class RankedRebalance(Strategy):
    def __init__(self, calendar=None):
        self.executor = TargetWeightExecutor(RebalanceConfig(
            schedule=RebalanceSchedule.fixed_n_sessions(2),
            calendar=calendar,
            data_frequency="daily",
            max_single_weight=0.6,
        ))
        self.decisions = []

    def on_data(self, timestamp, data, context, broker):
        targets = {
            asset: round(bar["signals"]["prediction"] * 0.8, 2)
            for asset, bar in data.items()
        }
        orders = self.executor.execute(targets, data, broker, timestamp=timestamp)
        if orders:
            self.decisions.append((timestamp, targets, len(orders)))


def run(category, first_asset, calendar=None):
    prices = load_example_prices(category)
    signals = prices.select("timestamp", "asset").with_columns(
        pl.when(pl.col("asset") == first_asset)
        .then(0.7).otherwise(0.3).alias("prediction")
    )
    strategy = RankedRebalance(calendar)
    result = Engine(DataFeed(prices_df=prices, signals_df=signals), strategy,
        BacktestConfig(initial_cash=100_000, calendar=calendar, timezone="America/New_York", enforce_sessions=True)).run()
    print(f"{category}: decisions={len(strategy.decisions)}, fills={len(result.fills)}")
    first = strategy.decisions[0]
    print(f"  first target: {first[0]:%Y-%m-%d} {first[1]}")
    print(f"  first fills: {[f.asset for f in result.fills[:2]]} at {[f.timestamp.strftime('%Y-%m-%d') for f in result.fills[:2]]}")

print("ml4t-backtest " + version("ml4t-backtest"))
run("equity", "AAPL", "NYSE")
run("etf", "SPY", "NYSE")
```

<!-- ml4t-doc-output: tutorial-multiasset-equity-etf -->
```text
ml4t-backtest {package_version}
equity: decisions=5, fills=6
  first target: 2024-01-02 {'AAPL': 0.56, 'MSFT': 0.24}
  first fills: ['AAPL', 'MSFT'] at ['2024-01-03', '2024-01-03']
etf: decisions=5, fills=5
  first target: 2024-01-02 {'SPY': 0.56, 'TLT': 0.24}
  first fills: ['SPY', 'TLT'] at ['2024-01-03', '2024-01-03']
```

`TargetWeightExecutor.execute()` turns the target dictionary into orders and
keeps the schedule state. `result.to_fills_dataframe()` records actual fills;
`strategy.decisions` records the targets used when orders were submitted. The
number of fills differs across the two panels because later price changes
produce different rounded share adjustments.

## Futures with different close times

The synthetic ES bar closes at 21:00 UTC and ZC at 21:15 UTC. Exact-timestamp
iteration therefore gives two partial callbacks per date. The observer below
records that default behavior without submitting orders. The rebalance run
uses the explicit `session_date` column, so both bars enter one decision at
21:15. The first ES and ZC orders fill on their own bars the next session.
The example specifies a $50 point multiplier and per-contract margin for each
synthetic contract. It does not claim that one exchange calendar covers both
products; the supplied session labels define this example's grouping.

<!-- ml4t-doc-test: tutorial-multiasset-future -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine
from ml4t.backtest import RebalanceConfig, RebalanceSchedule, Strategy, TargetWeightExecutor
from ml4t.backtest.example_data import load_example_prices


class Observe(Strategy):
    def __init__(self):
        self.events = []

    def on_data(self, timestamp, data, context, broker):
        self.events.append((timestamp, sorted(data)))


class FuturesRebalance(Strategy):
    def __init__(self):
        self.executor = TargetWeightExecutor(RebalanceConfig(
            schedule=RebalanceSchedule.fixed_n_sessions(2),
            data_frequency="daily", timezone="UTC", max_single_weight=0.6,
        ))
        self.decisions = []

    def on_data(self, timestamp, data, context, broker):
        targets = {asset: round(bar["signals"]["prediction"] * 0.8, 2)
                   for asset, bar in data.items()}
        orders = self.executor.execute(targets, data, broker, timestamp=timestamp)
        if orders:
            self.decisions.append((timestamp, targets))


prices = load_example_prices("future")
observer = Observe()
Engine(DataFeed(prices_df=prices), observer, BacktestConfig(initial_cash=1_000_000)).run()
signals = prices.select("timestamp", "asset").with_columns(
    pl.when(pl.col("asset") == "ES").then(0.7).otherwise(0.3).alias("prediction")
)
strategy = FuturesRebalance()
result = Engine(
    DataFeed(prices_df=prices, signals_df=signals, session_col="session_date"),
    strategy,
    BacktestConfig(initial_cash=1_000_000, allow_leverage=True, timezone="UTC"),
    contract_specs={
        "ES": ContractSpec("ES", AssetClass.FUTURE, multiplier=50, margin=12000),
        "ZC": ContractSpec("ZC", AssetClass.FUTURE, multiplier=50, margin=10000),
    },
).run()
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"exact events on first date: {[assets for ts, assets in observer.events if ts.day == 2]}")
print(f"session decision: {strategy.decisions[0][0]:%Y-%m-%d %H:%M} {strategy.decisions[0][1]}")
print(f"first fills: {[(f.asset, f.timestamp.strftime('%Y-%m-%d %H:%M')) for f in result.fills[:2]]}")
print(f"filled orders: {len(result.fills)}")
```

<!-- ml4t-doc-output: tutorial-multiasset-future -->
```text
ml4t-backtest {package_version}
exact events on first date: [['ES'], ['ZC']]
session decision: 2024-01-02 21:15 {'ES': 0.56, 'ZC': 0.24}
first fills: [('ES', '2024-01-03 21:00'), ('ZC', '2024-01-03 21:15')]
filled orders: 2
```

A session decision cannot fill against the earlier 21:00 bar. The feed also
rejects a session missing ES or ZC before the run starts. With the default
exact-timestamp feed, do not infer a zero target from an asset absent at the
current timestamp. See [Data Feed](../user-guide/data-feed.md#daily-decisions-across-different-close-times)
for the session input contract.

## USD-quoted FX pairs

The EURUSD and GBPUSD panels use UTC daily timestamps. Both prices are USD per
unit of base currency, and the contract specifications declare USD settlement.
No spot FX exchange calendar is enforced in this example. The schedule counts
the observed daily sessions and still sends orders for next-bar execution.

<!-- ml4t-doc-test: tutorial-multiasset-fx -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine
from ml4t.backtest import RebalanceConfig, RebalanceSchedule, Strategy, TargetWeightExecutor
from ml4t.backtest.example_data import load_example_prices


class FXRebalance(Strategy):
    def __init__(self):
        self.executor = TargetWeightExecutor(RebalanceConfig(
            schedule=RebalanceSchedule.fixed_n_sessions(2),
            data_frequency="daily", timezone="UTC", max_single_weight=0.6,
        ))
        self.decisions = []

    def on_data(self, timestamp, data, context, broker):
        targets = {asset: round(bar["signals"]["prediction"] * 0.8, 2)
                   for asset, bar in data.items()}
        orders = self.executor.execute(targets, data, broker, timestamp=timestamp)
        if orders:
            self.decisions.append((timestamp, targets))


prices = load_example_prices("fx")
signals = prices.select("timestamp", "asset").with_columns(
    pl.when(pl.col("asset") == "EURUSD").then(0.7).otherwise(0.3).alias("prediction")
)
strategy = FXRebalance()
result = Engine(
    DataFeed(prices_df=prices, signals_df=signals), strategy,
    BacktestConfig(initial_cash=100_000, timezone="UTC"),
    contract_specs={
        asset: ContractSpec(asset, AssetClass.FOREX, currency="USD")
        for asset in ("EURUSD", "GBPUSD")
    },
).run()
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"first target: {strategy.decisions[0][0]:%Y-%m-%d} {strategy.decisions[0][1]}")
print(f"first fills: {[(f.asset, f.timestamp.strftime('%Y-%m-%d')) for f in result.fills[:2]]}")
print(f"filled orders: {len(result.fills)}")
```

<!-- ml4t-doc-output: tutorial-multiasset-fx -->
```text
ml4t-backtest {package_version}
first target: 2024-01-02 {'EURUSD': 0.56, 'GBPUSD': 0.24}
first fills: [('EURUSD', '2024-01-03'), ('GBPUSD', '2024-01-03')]
filled orders: 8
```

The [Rebalancing](../user-guide/rebalancing.md) guide covers target sizing,
missing-price policies, and schedule metadata. The [Account Policies](../user-guide/accounts.md)
guide covers cash and margin settings used by these variants.

## In the book

Chapter 17, Section 17.7, [Comparing allocator performance](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/17_portfolio_construction/README.md),
and [notebook 08, Library comparison](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/17_portfolio_construction/08_library_comparison.ipynb)
compare portfolio construction workflows. [Chapter 16's futures notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/02_futures_backtesting.ipynb)
adds contract and overnight-session assumptions; the [FX case-study backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/case_studies/fx_pairs/13_backtest.ipynb)
applies targets to a larger prediction stream.
