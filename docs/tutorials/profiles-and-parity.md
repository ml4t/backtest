# Compare profile settings and bounded parity evidence

A profile fills many configuration fields at once. Inspect it before comparing
results or changing settings. The first example reads five named profiles from
the installed package. It reports configuration only; it does not execute an
external backtester. `zipline_strict` names the library's documented comparison
protocol, which differs from Zipline Reloaded's native daily fill and cost
defaults. The [Profiles](../user-guide/profiles.md) reference lists the other
fields and the retained evidence for supported workloads.

## Inspect the profiles

<!-- ml4t-doc-test: tutorial-profiles-inspect -->
```python
from importlib.metadata import version

from ml4t.backtest import BacktestConfig

print("ml4t-backtest " + version("ml4t-backtest"))
for name in ("default", "vectorbt_strict", "backtrader_strict", "zipline_strict", "lean"):
    cfg = BacktestConfig.from_preset(name)
    print(f"{name}: {cfg.execution_mode.value}/{cfg.execution_price.value} "
          f"shares={cfg.share_type.value} leverage={cfg.allow_leverage}")
```

<!-- ml4t-doc-output: tutorial-profiles-inspect -->
```text
ml4t-backtest {package_version}
default: next_bar/open shares=integer leverage=False
vectorbt_strict: same_bar/close shares=fractional leverage=False
backtrader_strict: next_bar/open shares=integer leverage=False
zipline_strict: next_bar/open shares=integer leverage=False
lean: next_bar/open shares=integer leverage=True
```

The table shows why comparing two unmodified profiles does not isolate an
execution rule: timing, share precision, account policy, costs, and other
fields may differ. It also does not establish native framework equivalence.

## Change one fill assumption

The next run holds the bundled synthetic AAPL bars, strategy, share quantity,
capital, and same-bar timing fixed. It changes only `execution_price`: open
versus close. The January 2 open is $187.725 and the close is $188. The order
fills on that same bar in both runs. Same-bar execution may use information
from the bar that is unavailable at the chosen fill time; use it here only to
measure the setting. The [orders tutorial](orders-and-timing.md) shows the
causal next-bar alternative.

<!-- ml4t-doc-test: tutorial-profiles-one-setting -->
```python
from dataclasses import replace
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy
from ml4t.backtest.config import ExecutionPrice
from ml4t.backtest.example_data import load_example_prices


class BuyOnce(Strategy):
    def __init__(self):
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            broker.submit_order("AAPL", 10)
            self.submitted = True


prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
base = BacktestConfig(initial_cash=100_000, execution_mode=ExecutionMode.SAME_BAR,
                      execution_price=ExecutionPrice.OPEN)
close_fill = replace(base, execution_price=ExecutionPrice.CLOSE)
print("ml4t-backtest " + version("ml4t-backtest"))
for label, config in (("same-bar open", base), ("same-bar close", close_fill)):
    result = Engine(DataFeed(prices_df=prices), BuyOnce(), config).run()
    fill = result.fills[0]
    print(f"{label}: {fill.timestamp:%Y-%m-%d} 10 at {fill.price:.5f} "
          f"final={result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: tutorial-profiles-one-setting -->
```text
ml4t-backtest {package_version}
same-bar open: 2024-01-02 10 at 187.72500 final=100082.75
same-bar close: 2024-01-02 10 at 188.00000 final=100080.00
```

The fill-price change is $0.275 per share. Ten shares cost $2.75 more in the
close-fill run, which accounts for its $2.75 lower final equity. No framework
is run here, so these two lines support only a local setting comparison.

## Read the retained comparison correctly

The [retained real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json)
used five frozen workloads: ETF allocation, CME futures, crypto perpetual
funding, USD-quoted FX allocation, and a US equity panel. Its
[generated report](../user-guide/profiles.md#real-strategy-audit) recorded
17 passing required framework-workload pairs and eight unsupported pairs
for the engine source digest stored with that evidence. Earlier engine changes
in this development branch made that digest stale; the parity-claim gate must
pass again before these rows can support a new release claim. The shared
targets were frozen before either engine ran. Those comparisons covered
ordered fills, supported valuation timestamps, and terminal values under
their declared profiles and canonical precision. They disabled transaction
costs and position rules, so even refreshed results will not establish
equivalence for those production overlays or for every asset class.

The [synthetic scenario results](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json)
check isolated conventions for pinned VectorBT, Backtrader, and Zipline
versions. The [large-scale results](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json)
check a 250-asset, 5,040-session workload and state which surfaces are native or
reconstructed. Neither expands the real-strategy claim to arbitrary inputs.
LEAN's supported daily US-equity protocol and separate case-study evidence
are described in the [Profiles](../user-guide/profiles.md) reference.

For a new framework comparison, freeze one input bundle and target stream,
select the matching profile and native framework configuration, then compare
only fields both sides expose. Record framework versions, data digests,
profile overrides, supported result surfaces, and the first differing record.
The [validation methodology](https://github.com/ml4t/backtest/blob/main/validation/METHODOLOGY.md)
describes the comparison process and canonical precision.

## In the book

Chapter 16, [Framework parity and engine divergence](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/README.md),
includes [notebook 07, Engine divergence anatomy](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/07_engine_divergence_anatomy.ipynb),
which changes one configuration field at a time, and [notebook 16, Case-study
LEAN parity](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/16_case_study_lean_parity.ipynb),
which reports the bounded cross-framework audit.
