# Bundled example data

The five price panels shipped with `ml4t-backtest` are synthetic. Their symbols, dates,
prices, volumes, and futures close times were written for these examples; they are not
historical market observations. Each panel has nine dates and two assets. The files
ship in the wheel, so the examples run offline without credentials. Every snippet
below runs on its own and submits one buy and one close order for its named asset.

The `timestamp` column is the bar time; `asset` identifies the instrument.
`open`, `high`, `low`, `close`, and `volume` provide the execution and mark inputs.
The futures file also has a `session_date` column because its two contracts close
at 21:00 and 21:15 in this synthetic calendar. `ExampleRoundTrip` buys after the
first asset bar and closes after its fifth; default next-bar execution fills at
the following bar's open. These examples exercise data loading and accounting,
not a profitable trading rule.

The output includes the installed distribution version. The documentation check
substitutes that exact version and compares the remaining lines byte for byte.

## Equities

<!-- ml4t-doc-test: smoke-equity -->
```python
from importlib.metadata import version
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices

prices = load_example_prices("equity")
feed = DataFeed(prices_df=prices)
config = BacktestConfig(initial_cash=100000)
engine = Engine(feed, ExampleRoundTrip("AAPL", 10), config)
result = engine.run()
assert len(result.fills) == 2 and len(result.trades) == 1
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"equity: fills={len(result.fills)}, trades={len(result.trades)}")
print(f"final equity: ${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: smoke-equity -->
```text
ml4t-backtest {package_version}
equity: fills=2, trades=1
final equity: $100030.00
```

## ETFs

<!-- ml4t-doc-test: smoke-etf -->
```python
from importlib.metadata import version
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices

prices = load_example_prices("etf")
feed = DataFeed(prices_df=prices)
config = BacktestConfig(initial_cash=100000)
engine = Engine(feed, ExampleRoundTrip("SPY", 10), config)
result = engine.run()
assert len(result.fills) == 2 and len(result.trades) == 1
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"etf: fills={len(result.fills)}, trades={len(result.trades)}")
print(f"final equity: ${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: smoke-etf -->
```text
ml4t-backtest {package_version}
etf: fills=2, trades=1
final equity: $100030.00
```

## Futures

ES uses a $50 point multiplier and an explicit 21:00 bar close. The decision waits for the later ZC close.

<!-- ml4t-doc-test: smoke-future -->
```python
from importlib.metadata import version
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices
from ml4t.backtest import AssetClass, ContractSpec

prices = load_example_prices("future")
feed = DataFeed(prices_df=prices, session_col="session_date")
config = BacktestConfig(initial_cash=200000, allow_leverage=True)
spec = ContractSpec(symbol="ES", asset_class=AssetClass.FUTURE, multiplier=50, margin=12000)
engine = Engine(feed, ExampleRoundTrip("ES", 1), config, contract_specs={"ES": spec})
result = engine.run()
assert len(result.fills) == 2 and len(result.trades) == 1
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"future: fills={len(result.fills)}, trades={len(result.trades)}")
print(f"final equity: ${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: smoke-future -->
```text
ml4t-backtest {package_version}
future: fills=2, trades=1
final equity: $201000.00
```

## FX

The pair is USD quoted, so this example does not need a currency conversion feed.

<!-- ml4t-doc-test: smoke-fx -->
```python
from importlib.metadata import version
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices
from ml4t.backtest import AssetClass, ContractSpec

prices = load_example_prices("fx")
feed = DataFeed(prices_df=prices)
config = BacktestConfig(initial_cash=100000)
spec = ContractSpec(symbol="EURUSD", asset_class=AssetClass.FOREX)
engine = Engine(feed, ExampleRoundTrip("EURUSD", 10000), config, contract_specs={"EURUSD": spec})
result = engine.run()
assert len(result.fills) == 2 and len(result.trades) == 1
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"fx: fills={len(result.fills)}, trades={len(result.trades)}")
print(f"final equity: ${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: smoke-fx -->
```text
ml4t-backtest {package_version}
fx: fills=2, trades=1
final equity: $100050.00
```

## Crypto perpetuals

The contract uses the futures accounting convention; a later tutorial adds funding payments.

<!-- ml4t-doc-test: smoke-crypto-perp -->
```python
from importlib.metadata import version
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices
from ml4t.backtest import AssetClass, ContractSpec

prices = load_example_prices("crypto_perp")
feed = DataFeed(prices_df=prices)
config = BacktestConfig(initial_cash=100000, allow_leverage=True)
spec = ContractSpec(symbol="BTC-PERP", asset_class=AssetClass.FUTURE, margin=4000)
engine = Engine(feed, ExampleRoundTrip("BTC-PERP", 1), config, contract_specs={"BTC-PERP": spec})
result = engine.run()
assert len(result.fills) == 2 and len(result.trades) == 1
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"crypto_perp: fills={len(result.fills)}, trades={len(result.trades)}")
print(f"final equity: ${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: smoke-crypto-perp -->
```text
ml4t-backtest {package_version}
crypto_perp: fills=2, trades=1
final equity: $100700.00
```

## Use longer data

For US equities and ETFs, [Alpha Vantage's daily API](https://www.alphavantage.co/documentation/)
offers a free compact history after registering for an API key. The offline examples above
need no key. The [ECB exchange-rate series](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.et.html)
provides public FX reference rates, not executable OHLC bars. [Binance's public archive](https://github.com/binance/binance-public-data/blob/master/README.md)
provides spot and futures klines for crypto research. [CME daily settlements](https://www.cmegroup.com/market-data/daily-settlements.html)
show contract marks; check CME's data access terms before building a historical
futures panel. Normalize vendor timestamps, corporate actions, currency units,
contract multipliers, and missing sessions before substituting any of these
sources for the synthetic panels.

## In the book

Chapter 16, Section 16.3, [Futures backtesting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/02_futures_backtesting.ipynb) extends the small synthetic futures panel to a longer contract history. [FX pairs backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/case_studies/fx_pairs/13_backtest.ipynb) shows how a research prediction stream becomes aligned feed input.
