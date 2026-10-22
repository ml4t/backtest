# Run a futures variation account

Use an explicit `ContractSpec(accounting_method="futures_variation")` to trade
contracts without exchanging their full notional through cash. Keep ordinary
equities and futures in the same native `Engine` account. The existing default,
`signed_notional`, remains available for strategies that depend on it.

This guide describes the next-version source boundary. It is absent from the
published 0.1.12 package and preserved 0.1.13 candidate. Build a wheel from the
revision containing this page. The examples require no Zipline installation,
network access or ML4T Data dependency.

## Pay a declared settlement

Declare raw contract prices, multiplier, currency and margin. Use a margin-enabled
account and choose `settlement_policy="scheduled"` for explicit settlement events.
The example pays $500 of variation on the last observation. Before payment, the
same $500 contributes to NAV as unsettled P&L. The $15,000 initial margin requirement
encumbers funds; it is neither a fee nor a deduction from cash.

<!-- ml4t-doc-test: futures-scheduled-account -->
```python
from datetime import UTC, datetime, timedelta
from tempfile import TemporaryDirectory
import polars as pl
from ml4t.backtest import AssetClass, BacktestConfig, BacktestResult
from ml4t.backtest import ContractSpec, Strategy, run_backtest
from ml4t.backtest.config import CommissionType

times = [datetime(2025, 3, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(4)]
prices = pl.DataFrame({
    "timestamp": times, "asset": ["ESM25"] * 4,
    "close": [4000.0, 4000.0, 4010.0, 4010.0],
})
settlements = pl.DataFrame([{
    "timestamp": times[3], "available_at": times[2], "asset": "ESM25",
    "settlement_price": 4010.0, "currency": "USD",
    "source": "declared-example", "event_id": "settlement-1", "revision_id": "r1",
}])

class Buy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp == times[0]:
            broker.submit_order("ESM25", 1)

result = run_backtest(
    prices, Buy(),
    config=BacktestConfig(
        calendar=None, initial_cash=100000, allow_leverage=True,
        commission_type=CommissionType.PER_TRADE, commission_per_trade=2,
        slippage_rate=0, fixed_margin_schedule={"ESM25": (15000, 12000)},
    ),
    contract_specs={"ESM25": ContractSpec(
        "ESM25", asset_class=AssetClass.FUTURE, multiplier=50, tick_size=0.25,
        currency="USD", accounting_method="futures_variation",
        settlement_policy="scheduled",
    )},
    settlement_events=settlements, holdings_sampling="bar",
)
account = result.to_portfolio_state_dataframe()
assert account["cash"].to_list() == [100000, 99998, 99998, 100498]
assert account["equity"].to_list() == [100000, 99998, 100498, 100498]
assert account["positions_value"].to_list() == [0, 0, 500, 0]
assert account["net_exposure"][-1] == 200500
assert account["initial_margin_required"][-1] == 15000
assert account["available_funds"][-1] == 85498
assert result.to_futures_settlements_dataframe()["cash_delta"].to_list() == [500]
with TemporaryDirectory() as directory:
    result.to_parquet(directory)
    restored = BacktestResult.from_parquet(directory)
    assert restored.to_portfolio_state_dataframe().equals(account)
```

`Engine`, `Engine.from_config` and `run_backtest` accept the same
`settlement_events` frame. Its clocks must be physical instants compatible with the
feed, with `available_at <= timestamp`. Payment uses positions held when the event
is processed, before that observation's callbacks and fills. A settlement price
changes cash and the variation basis; it does not create an execution quote.

Every event needs the fields shown above. Currency must match the instrument;
the single cash account does not convert currencies. An identical event retry is
deduplicated by `(source, event_id)`. Conflicting revisions are rejected; supply a
separate, explicitly dated forward event instead of replacing an applied event.

## Keep callback strategies with mark-driven payout

For the tested Zipline payout method, choose `settlement_policy="on_mark"`.
The adapter retains ordinary callback orders and projects the same native result.
This synthetic dated contract uses three actual calendar-close instants.

<!-- ml4t-doc-test: futures-mark-callbacks -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import AssetClass, ContractSpec, DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Future
from ml4t.backtest.compat.zipline.finance import commission, slippage

contract = Future(
    "example:ESM25", symbol="ESM25", legacy_sid=7,
    root_symbol="ES", multiplier=50, tick_size=0.25,
)
feed = DataFeed(
    prices_df=pl.DataFrame({
        "timestamp": [datetime(2025, 3, day, 22, tzinfo=UTC) for day in [3, 4, 5]],
        "asset": [contract.key] * 3, "close": [4000.0, 4000.0, 4010.0],
        "volume": [1000.0] * 3,
    }),
    feed_spec={
        "time_contract_version": 2,
        "calendar": "us_futures", "timestamp_semantics": "event_time",
        "data_frequency": "daily", "bar_label": "right",
        "bar_grid_anchor": "session_open", "partial_bar_policy": "include",
    },
)
def initialize(context):
    context.set_benchmark(None)
    context.set_commission(us_futures=commission.PerFutureTrade(2))
    context.set_slippage(us_futures=slippage.NoSlippage())
    context.sent = False

def handle_data(context, data):
    if not context.sent:
        context.order(contract, 1)
        context.sent = True

performance = run_algorithm(
    feed=feed, assets=[contract], initialize=initialize, handle_data=handle_data,
    capital_base=100000,
    contract_specs={contract.key: ContractSpec(
        contract.key, asset_class=AssetClass.FUTURE, multiplier=50, tick_size=0.25,
        currency="USD", accounting_method="futures_variation", settlement_policy="on_mark",
    )},
)
assert performance["cash"].to_list() == [100000, 99998, 100498]
assert performance["portfolio_value"].to_list() == [100000, 99998, 100498]
assert performance["positions_value"].to_list() == [0, 0, 0]
native_result = performance.attrs["backtest_result"]
assert native_result.account_method == "per_instrument"
```

Mark-driven payout transfers variation on an observed mark before callbacks and
after fills. A fill first pays the entire previous position to its transaction
price, then establishes the resulting position's basis. Scheduled settlement
instead realizes the closed portion at a fill and leaves the remaining position's
variation unpaid. Account reads never create transfers. Entry, scale-in, partial
close and reversal fees are charged once through the existing fill ledger.
Submission prechecks, buying-power reservations and queue validation use the same
variation-basis fill rule. Legacy simple cash checks do not exchange futures
principal; variation instruments use their declared margin requirements.

The callback adapter uses its explicit unconstrained order-entry profile. It does
not establish parity with a broker's collateral rules. Use native `Engine` with
declared margin schedules for constrained entry and scheduled settlement.

## Inspect the account and migrate results

The account is labeled `per_instrument`; `instrument_accounting` identifies opted-in
contracts, settlement policy, multiplier and currency. Other instruments retain
signed-notional accounting. Portfolio observations retain:

| Field | Meaning |
|---|---|
| `gross_exposure`, `net_exposure` | Absolute and signed economic notional, including futures multipliers. |
| `positions_value` | Equity liquidation value plus unsettled futures P&L; excludes futures principal. |
| `futures_unsettled_pnl` | Variation not yet transferred into cash. |
| `initial_margin_required`, `maintenance_margin_required` | Requirements evaluated from actual observed positions and marks. |
| `available_funds` | NAV minus unpaid corporate receivables, initial margin, settlement holds and cash buffer. |
| `settlement_holds`, `cash_buffer` | Existing sale-proceeds holds and configured cash reserve. |
| `margin_breach` | Whether observed NAV is below maintenance requirements. No invented liquidation fill follows. |

Holdings retain both `signed_notional` and `liquidation_value`. A complete futures
payout leaves liquidation value zero even while exposure remains. Schema 6 portable
results retain these actual observations and the required, content-bound
`futures_settlements` table. The table separates scheduled/mark/fill transfers from
commissions and identifies the actual fill when a transfer occurs at execution.
Existing schema 2 through 5 artifacts remain readable with their original methods.
Corporate receivables and prepared benchmarks can coexist in the same artifact.

The captured reference tests in
[`test_zipline_futures_variation.py`](https://github.com/ml4t/backtest/blob/fcde877fd686d599d4c4283d3122ddc9a77976d2/tests/test_zipline_futures_variation.py)
compare eight original Zipline Reloaded 3.2.0 runs: long, short, close and
scale-in/partial-close/reversal, each with zero and fixed per-future fees. Cash,
NAV, position basis, marks and fills match exactly for those declared inputs.
The fixture also passes through ML4T Data's actual snapshot reader. These tests
do not qualify an arbitrary contract chain, roll policy or exchange settlement feed.

Use [dated-contract rolls and lifecycle policies](dated-futures.md) to execute
causal selections through raw contract orders. Keep those decisions separate from
this cash method. The
continuous-root CME performance workload lacks a dated chain/roll map and remains
outside the Zipline comparison. No new speed claim follows from this accounting
interface. See [migration assumptions](migrate-from-zipline.md#match-ledger-and-corporate-action-assumptions)
and [raw corporate actions](corporate-actions.md).
