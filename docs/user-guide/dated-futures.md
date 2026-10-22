# Execute dated futures rolls and lifecycle boundaries

Trade raw dated contracts using their stable IDs. Use Data's continuous series for
research and its point-in-time selections to decide which contract to trade.
Backtest routes actual close/open orders, applies costs and margin, and records
partial or rejected legs. An adjusted continuous price is never an execution quote.

These interfaces require a wheel built from the revision containing this guide.
They are absent from published Backtest 0.1.12 and the preserved 0.1.13 release
candidate. No Zipline installation is required for production use.

## Roll through the native broker

`broker.roll_position(old_asset, new_asset, roll_id=..., decision_at=...,
available_at=..., new_quantity=..., selection=...)` submits two ordinary market
orders under one roll ID. Both contracts need explicit future `ContractSpec`s.
`decision_at` defaults to the current clock. A declared `available_at` cannot
follow the decision, and the decision cannot follow submission. Supply the actual
selection row through `selection` to retain its source facts with the roll.

The default replacement quantity equals the signed old position. Different
multipliers require an explicit `new_quantity`; Backtest does not infer a hedge
ratio. Resolve pending orders on both contracts before rolling. Repeating the same
roll ID returns its existing orders without paying fees twice. A conflicting reuse
raises an error.

The legs are not atomic. A closing fill can coexist with a pending or rejected
opening order, and either leg can fill partially under ordinary volume limits.
Read `result.to_rolls_dataframe()` for actual quantities, order/fill IDs, fees,
submission/execution clocks and completion state. Schema 7 exports retain this
table and validate it against actual fills on import. Existing schema 2 through 6
readers retain their original accounting definitions.

## Close or roll before a declared lifecycle boundary

Pass `FuturesLifecyclePolicy` declarations to `RuntimeServices(futures_lifecycle=...)`.
Each policy identifies one controlling physical boundary for a dated contract:
`first_notice`, `last_trade`, `auto_close` or `expiration`. Required fields are
`event_id`, `asset`, `effective_at`, `available_at`, `reason`, `source` and
`revision_id`. Add `replacement_asset` for a roll, and `new_quantity` only when
an explicit replacement size is needed.

Choose the boundary upstream. A source date does not specify a liquidation instant.
For example, closing before first notice may mean the preceding session's close;
that choice requires the contract's calendar and your delivery policy. Preserve the
source dates and that conversion alongside the physical policy. Unknown terms,
missing publication times and bare dates cannot certify an executable lifecycle.
No automatic delivery, exercise, FX conversion or exchange settlement is inferred.

At the boundary the existing clock scheduler cancels pending orders on the old
contract and submits its closing order or paired roll. A native trading control
then prevents increases or reversals of the old contract's exposure. The callback
has no execution quote: even immediate-fill strategies wait for an actual price
observation. Missing quotes leave pending orders. A boundary after captured data
does not close an open position. A boundary inside an excluded execution window
is rejected; choose an executable policy boundary explicitly.

The following complete example rolls once and later closes the replacement.
The old leg earns $1,000 and the replacement earns $1,000. Four $2 commissions
leave cash and NAV at $101,992. Save policy declarations alongside the result;
the roll table records roll decisions and actual leg outcomes.

<!-- ml4t-doc-test: dated-futures-roll-and-notice -->
```python
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import polars as pl
from ml4t.backtest import AssetClass, BacktestConfig, BacktestResult, ContractSpec
from ml4t.backtest import DataFeed, Engine, FuturesLifecyclePolicy, RuntimeServices, Strategy
from ml4t.backtest.config import CommissionType

times = [datetime(2025, 3, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(6)]
prices = pl.DataFrame({
    "timestamp": [t for t in times for _ in range(2)],
    "asset": ["capture:ESH25", "capture:ESM25"] * 6,
    "close": [4000., 4010., 4000., 4010., 4010., 4020.,
              4020., 4030., 4030., 4040., 4040., 4050.],
})
policy = FuturesLifecyclePolicy(
    event_id="close-ESM25-r1", asset="capture:ESM25",
    effective_at=times[5] - timedelta(hours=1), available_at=times[0],
    reason="auto_close", source="declared-example", revision_id="r1",
)
class Roll(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp == times[0]:
            broker.submit_order("capture:ESH25", 1)
        if timestamp == times[2]:
            broker.roll_position(
                "capture:ESH25", "capture:ESM25", roll_id="selection-1",
                decision_at=times[2], available_at=times[1],
                selection={"source": "declared-example", "revision_id": "r1"},
            )
result = Engine(
    DataFeed(prices_df=prices, feed_spec={
        "time_contract_version": 2, "calendar": None,
        "timestamp_semantics": "event_time",
    }), Roll(),
    BacktestConfig(
        calendar=None, initial_cash=100000, allow_leverage=True,
        commission_type=CommissionType.PER_TRADE, commission_per_trade=2,
        slippage_rate=0,
    ),
    runtime=RuntimeServices(futures_lifecycle=[policy]),
    contract_specs={asset: ContractSpec(
        asset, asset_class=AssetClass.FUTURE, multiplier=50, currency="USD",
        margin=15000, accounting_method="futures_variation", settlement_policy="on_mark",
    ) for asset in ["capture:ESH25", "capture:ESM25"]},
    holdings_sampling="bar",
).run()
assert [(f.timestamp, f.price) for f in result.fills] == [
    (times[1], 4000), (times[3], 4020), (times[3], 4030), (times[5], 4050),
]
assert result.to_portfolio_state_dataframe()["cash"][-1] == 101992
assert result.to_equity_dataframe()["equity"][-1] == 101992
assert result.to_rolls_dataframe()["roll_state"].to_list() == ["completed"] * 2
with TemporaryDirectory() as directory:
    result.to_parquet(directory)
    Path(directory, "lifecycle-policy.json").write_text(
        json.dumps(asdict(policy), default=lambda t: t.isoformat())
    )
    restored = BacktestResult.from_parquet(directory)
    assert restored.to_rolls_dataframe().equals(result.to_rolls_dataframe())
```

Use [futures variation accounting](futures-accounting.md) for cash, settlement,
margin and mixed-asset definitions. The [account exports](results.md) also provide
analysis-ready observations for Diagnostic; migration can retain callback strategies
while using that native result pipeline.

## Consume Data selections without changing their meaning

The integration tests use actual ML4T Data
`0.2.2.dev21+gc3b9db80a`, source revision
`c3b9db80a66397420fb27adeccc762559dd41e07`, with published Specs 0.2.0.
This is a tested source identity, not a claim that an earlier published Data wheel
contains that preparation API.

Call `ml4t.data.futures.prepare_futures_inputs` at each decision's actual `as_of`
cutoff. Consume its `normalized_feed_spec`, normalized contract units and stable
IDs. Backtest account keys are strings; the tested input explicitly maps integer
source IDs 17 and 23 to `capture:17` and `capture:23` on both prices and orders.
Do not silently substitute tickers or cast identities in only one table.

A `HighestVolumeRoll` selection uses the preceding observed session. The tested
chain selects 17 on July 8 from July 7 volume, then 23 on July 9 from July 8 volume.
An open-interest value published on July 10 cannot select a contract on July 9.
The selection row's `decision_at` and `metric_available_at` feed the corresponding
broker arguments. Its paired old/new marks describe source observations; only
later raw quotes determine fills.

The runnable integration workflow is
[`test_data_dated_rolls.py`](https://github.com/ml4t/backtest/blob/0ab0e577f1d7dc0031249dd51fc8ae45e12355ce/tests/test_data_dated_rolls.py).
With that Data build installed in a Backtest source checkout, run
`uv run pytest tests/test_data_dated_rolls.py -o addopts='' -q`.
It tests both the original inputs and changed future volume/open interest, checks
identical earlier selections and fills, and reconciles cash independently.
Calendar-free Data futures preparation is not qualified by this workflow; it uses
explicit `CME_Equity` event instants.

## Migrate an original Zipline futures workflow

Original `continuous_future(root, offset=0, roll="volume", adjustment=None)`
returns a research handle. The Backtest callback adapter now provides the same
lookup through a registered `PreparedFutures` source. Querying its selected
contract does not itself roll a held position. Retain the selection operation
in Data and explicitly call `roll_position` when the strategy intends to move
its position. The
[callback adapter](migrate-from-zipline.md) can retain the surrounding strategy;
it does not turn an adjusted series into a tradable instrument.

The published `zipline-reloaded==3.2.0` reference uses an explicit `us_futures`
calendar, raw individual contracts, zero costs and its daily auto-close behavior.
Its volume chain changes from 17 to 23 on July 9; its calendar chain changes on
July 10. Holding 17 does not open 23 automatically. Original auto-close produces
a closing transaction for 17 on July 10. The native explicit raw-quote close
matches its complete fills, cash and NAV on those captured inputs.

Reproduce the old workflow with
[`capture_zipline_dated_reference.py`](https://github.com/ml4t/backtest/blob/c18834e4da7fc288cd022a00b9aca71764b551fa/validation/capture_zipline_dated_reference.py)
in a separate environment containing Zipline 3.2.0. Give the script a new output
directory. Compare the new workflow with
[`test_zipline_dated_reference.py`](https://github.com/ml4t/backtest/blob/0ab0e577f1d7dc0031249dd51fc8ae45e12355ce/tests/test_zipline_dated_reference.py).
These are correctness captures, not performance measurements.

Original expiry cleanup can synthesize a closing transaction through its data
portal. The native policy above requires an actual executable quote. Missing
quotes therefore remain an explicit method difference, with an observable pending
order rather than an invented liquidation. The tested chain does not establish
parity for arbitrary contract metadata, adjustments, delivery or exchange expiry
settlement rules.

## Retain continuous-futures callback reads

Register `ContinuousFuture(key, root_symbol=..., offset=0, roll_style="volume",
adjustment="mul")` and `PreparedFutures(handle, provider, contracts, exchange,
source_id)` through `run_algorithm(continuous_futures=[source], ...)`.
`contracts` maps original stable source IDs to declared dated `Future` objects
that also appear in `assets`. The provider calls Data's `prepare_futures_inputs`
with the actual requested `as_of`. The adapter checks the physical publication
cutoffs, selected front, explicit ID mapping and normalized contract units.

Inside callbacks, use the familiar API:

```python
from ml4t.backtest.compat.zipline.api import continuous_future

def initialize(context):
    context.cf = continuous_future("TEST", offset=0, roll="volume", adjustment="mul")

def handle_data(context, data):
    contract = data.current(context.cf, "contract")
    chain = data.current_chain(context.cf)
    if contract is not None and data.can_trade(context.cf):
        price = data.current(context.cf, "price")
        history = data.history(context.cf, "price", 2, "1d")
```

`offset` counts dated maturities after the selected front. It is not a volume
rank. An exhausted chain returns `None` for `contract`, an empty chain, and NaN
for price. Order the actual returned `Future`; ordering the research handle is
an error. A changed research front alone leaves an existing position unchanged.

Continuous history must come from a named prepared source. Declare its research
keys with `RuntimeServices(history_assets={source_name: [handle.key]},
history_sources={source_name: provider})`, and map the requested frequency using
`run_algorithm(history_sources={"1d": source_name}, ...)`. These keys extend
only the read service. They add no observations or tradable instruments to the
execution feed. Undeclared keys and future or unpublished history are rejected.
The declared source must prepare the handle's requested roll and adjustment.

### Run the captured volume-roll migration

With the pinned Data source from [the offline installation recipe](offline-migration.md)
installed alongside this Backtest source build, run the complete synthetic example:

```bash
python -I examples/continuous_futures_migration.py ./continuous-example --adjustment mul
```

Use a new output directory for each `none`, `add` or `mul` run. The example
captures immutable Data inputs, calls Data at each decision cutoff, prepares
history through Data's adjustment APIs, runs the native callback engine, and
saves the actual result plus `workflow.json`. The report retains the input
snapshot ID, roll policy, adjustment, package versions, effective configuration,
callback chain/history and actual fills. No RNG is used.

The July 9 decision selects source ID 23 from July 8 volume. Raw current price
is 4,030. Its two-bar history is `[4010, 4030]` without adjustment and
`[4020, 4030]` with either adjustment in this example. A one-contract order
submitted then fills on July 10 at the actual dated quote 4,040. The research
series cannot supply its execution price.

| Input policy | Tested meaning and migration choice |
|---|---|
| `HighestVolumeRoll` | Uses preceding observed-session volume. Its matching selections are checked against the captured Zipline 3.2 example. It does not reproduce every Zipline grace-period or no-flip-back rule for arbitrary chains. |
| `CalendarRoll` | Data selects by expiration. To reproduce the captured original's July 10 auto-close-based selection, use actual first-notice facts and `FirstNoticeDateRoll(0, calendar="CME_Equity")`; that matching route is tested separately. |
| `roll_marks="previous_session"` | The example's default supplies the preceding common session's old/new marks to Data's adjustment methods. This reproduces the captured original's additive and multiplicative histories. |
| `roll_marks="same_session"` | Uses Data's `build_roll_events` switch-session paired marks. It is tested as a separate method; multiplicative history can differ from the original convention. |

The preparation recipe is for the declared single-root daily synthetic input.
Intraday aggregation, another chain's lifecycle rules, or a different quote scale
requires its own Data preparation and matched validation. Keep executable
lifecycle policies separate from research selection, as in the native example
above. The adapter does not infer delivery or fabricate an expiry fill.
