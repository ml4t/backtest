# User Guide

Use this guide to build a backtest from prepared market data and strategy decisions, then inspect what the engine actually executed. The [quickstart](../getting-started/quickstart.md) gives you a complete first run. The topic pages below explain the choices you make when adapting that run to your own data and strategy.

The code in the [Getting Started tutorials](../getting-started/quickstart.md) runs from the installed package against bundled synthetic data. The topic pages describe individual interfaces and settings. When a topic page shows only a method call, use its linked tutorial for a complete program and expected output. The [API reference](../api/index.md) gives the signatures and source docstrings for the released package.

## Find the task you need

| Task | Start here | Complete example |
|---|---|---|
| Prepare bars, signals, context, and timestamps | [Data Feed](data-feed.md) | [Bundled example data](../tutorials/data.md) |
| Write decisions and carry state between callbacks | [Strategies](strategies.md) and [Stateful Strategies](stateful-strategies.md) | [First backtest](../getting-started/quickstart.md) and [Risk and State](../tutorials/risk-and-state.md) |
| Choose order types and determine when fills can occur | [Order Types](orders.md) and [Execution Semantics](execution-semantics.md) | [Orders and Timing](../tutorials/orders-and-timing.md) |
| Allocate across assets and deal with different daily close times | [Rebalancing](rebalancing.md) | [Multi-asset Rebalancing](../tutorials/multiasset-rebalancing.md) |
| Set cash, margin, contract, and sizing rules | [Account Policies](accounts.md) and [Configuration](configuration.md) | [Accounts and Constraints](../tutorials/accounts-and-constraints.md) |
| Model commission, slippage, impact, and funding | [Market Impact and Execution Costs](market-impact.md) | [Costs and Funding](../tutorials/costs-and-funding.md) |
| Add position exits and portfolio limits | [Risk Management](risk-management.md) | [Risk and State](../tutorials/risk-and-state.md) |
| Compare explicit execution settings | [Profiles](profiles.md) | [Profiles and Parity](../tutorials/profiles-and-parity.md) |
| Export fills, trades, equity, and portfolio state | [Results and Analysis](results.md) | [Results and Analysis](../tutorials/results-and-analysis.md) |

## Choose an asset example

All five bundled panels are synthetic. They test the library workflow offline; they do not estimate a strategy's live performance or reproduce a book case study. The [data page](../tutorials/data.md) runs a round trip for each panel and states its price and volume assumptions.

| Asset setting | Example | Setting to check before using your own data |
|---|---|---|
| Equity | [Equity round trip](../tutorials/data.md#equities) | Share precision, trading calendar, and corporate-action handling in the input |
| ETF | [ETF round trip](../tutorials/data.md#etfs) | Weight targets, rebalance timing, and cash available for the next fill |
| Futures | [Futures round trip](../tutorials/data.md#futures) | Contract multiplier, margin, and the timestamp of each market close |
| FX | [FX round trip](../tutorials/data.md#fx) | Pair quotation and the account currency used for valuation |
| Crypto perpetual | [Perpetual round trip](../tutorials/data.md#crypto-perpetuals) | Funding timestamps, rate sign, and the position held before each event |

The examples use declared settings rather than implied market defaults. Start with the [configuration guide](configuration.md) when you replace a bundled panel with another instrument or venue.

## From a book notebook to a library run

The [Book Guide](../book-guide/index.md) maps checked companion notebooks to the matching library workflows. Begin with the library quickstart, then compare the notebook's data, decision time, fill time, account rules, and costs with the settings in your run. Keep the original notebook's research question separate from the engine configuration. The book links point to a pinned companion revision, and each linked tutorial states what its notebook adds.

## Check a result

Inspect [fills](results.md#fills), [trades](results.md#trades-dataframe), and [equity](results.md#equity-dataframe) before interpreting summary metrics. A missing fill can mean that an order was rejected, an order type did not trigger, the next eligible bar was absent, or buying power was insufficient. The [result tutorial](../tutorials/results-and-analysis.md) exports these records and joins event timestamps back to the input data. The [diagnostic handoff](../tutorials/diagnostic-handoff.md) shows the optional analysis package used for downstream metrics.
