# Raw corporate actions

Use `DataFeed(corporate_actions=..., input_manifest=...)` or the same keywords on
`run_backtest` to apply declared events to raw prices. Backtest changes the existing
account, positions and pending orders. Data preparation belongs in ML4T Data;
adjusted historical series alone do not create dividend cash or payment entitlements.

This interface is part of the next-version source boundary. It is not present in
the published 0.1.12 package or the preserved 0.1.13 release candidate. Use a wheel
built from the source revision containing this page. The example needs only Backtest
and its normal dependencies; Zipline and ML4T Data are not required at runtime.

## Run a split followed by a dividend

All timestamps below are explicit UTC instants, so this example disables calendars.
The account buys ten shares for $1,000, receives a 2:1 split, earns a $20 dividend
on twenty shares, then receives the payment. Receivables contribute to NAV but
cannot be spent before payment.

<!-- ml4t-doc-test: raw-corporate-actions -->
```python
from datetime import UTC, datetime, timedelta
import polars as pl
from ml4t.backtest import BacktestConfig, ContractSpec, Strategy, run_backtest

times = [datetime(2025, 3, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(5)]
prices = pl.DataFrame({
    "timestamp": times, "asset": ["A"] * 5,
    "close": [100.0, 100.0, 50.0, 49.0, 49.0],
})
common = {
    "source": "declared-example", "revision_id": "r1", "asset": "A",
    "available_at": times[0], "currency": "USD", "share_basis": "pre_event",
}
events = pl.DataFrame([
    {**common, "event_id": "split", "action_kind": "split",
     "effective_at": times[2], "split_ratio": 2.0,
     "split_ratio_convention": "new_per_old"},
    {**common, "event_id": "dividend", "action_kind": "cash_dividend",
     "effective_at": times[3], "payment_at": times[4], "cash_per_share": 1.0},
])

class Buy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp == times[0]:
            broker.submit_order("A", 10)

result = run_backtest(
    prices, Buy(),
    config=BacktestConfig(calendar=None, initial_cash=10000,
                          commission_rate=0, slippage_rate=0),
    contract_specs={"A": ContractSpec("A", currency="USD")},
    corporate_actions=events, input_manifest={"price_basis": "raw"},
    holdings_sampling="bar",
)
account = result.to_portfolio_state_dataframe()
assert account["cash"].to_list() == [10000, 9000, 9000, 9000, 9020]
assert account["receivables_value"].to_list() == [0, 0, 0, 20, 0]
assert account["equity"].to_list() == [10000] * 5
assert result.to_actions_dataframe().filter(pl.col("operation") == "split")[
    "quantity_after"
].to_list() == [20]
assert result.to_entitlements_dataframe()["paid"].to_list() == [True]
print("20 shares, cash 9020, NAV 10000; one dividend paid")
```

<!-- ml4t-doc-output: raw-corporate-actions -->
```text
20 shares, cash 9020, NAV 10000; one dividend paid
```

## Event fields

Pass a Polars frame. Every event requires nonempty `source`, `event_id`,
`revision_id` and stable `asset`, plus `action_kind`, `effective_at`, a physical
`available_at`, and `currency` matching the instrument's explicit `ContractSpec`.
There is no implicit FX conversion. Source and destination instruments must use
the same currency for security delivery.

| Kind | Economic terms | Account effect |
|---|---|---|
| `split` | Positive `split_ratio`, `split_ratio_convention="new_per_old"` | Rebase quantities, basis and pending order prices. Original fills remain unchanged. |
| `cash_dividend`, `cash_distribution` | Nonnegative `cash_per_share`, `share_basis`, payment clock | Capture eligible shares once, then credit cash or debit a short liability at payment. |
| `stock_dividend`, `stock_distribution` | Positive `quantity_ratio`, stable `destination_asset`, `share_basis`, payment clock | Capture the security entitlement and deliver to the existing destination position. |
| `merger` | Nonnegative `quantity_ratio` and `cash_component`, destination when ratio is nonzero, `share_basis` | Remove the source position, cancel source orders and deliver the declared cash/security consideration. Further source orders are rejected. |

`quantity_ratio` is destination shares per source share. `cash_component` is cash
per source share. Native conversion carries residual source book basis into the
security leg and recognizes cash exceeding that basis as realized gain. A cash-only
conversion realizes the difference from source basis. These are simulation book
rules, not a tax-lot or statutory basis allocation service.

For a native split, fractional shares are retained. To receive cash instead, declare
`fractional_policy="cash_in_lieu"` and a positive `cash_in_lieu_price` in the new
share units. Partial orders adjust total and already-filled quantities independently;
`requested_quantity` and original fills retain their original units.

## Clocks and simultaneous events

The native `declared_terms` policy uses explicit `entitlement_at`, then `ex_date`,
then `entitlement_date`, then `effective_at` for distribution eligibility.
`zipline_3_2` uses `ex_at` in place of `entitlement_at`. A record date does not
replace an explicitly declared ex date. Payment uses `payment_at`, then
`payment_date`; a merger without a payment clock pays at its effective instant.

Civil `Date` labels require a declared calendar and a valid session. They apply
from UTC session midnight, before premarket callbacks. Physical instants retain
their declared instant and must match the feed's timezone awareness. With
`calendar=None`, use physical clocks for every clock actually used by the event.
No event can be published after its effective/eligibility instant.

At one clock, Backtest captures `pre_event` entitlements, applies splits and
conversions, captures `post_event` entitlements, then pays due obligations. Event
identities provide a deterministic order within each phase. The group is atomic:
an invalid resulting position restores cash, holdings, pending orders and journals.
Actions run before strategy callbacks and subsequent order execution at that clock.

An exact repeated revision applies once. Conflicting terms under the same revision
fail. A later publication may replace earlier terms only before the earliest
effective instant. Different revisions competing for the latest publication clock
are ambiguous.
Retroactive corrections require a separately declared forward event.

## Input basis and audit outputs

Declare `input_manifest={"price_basis": "raw"}` or pass an actual ML4T Data input
snapshot manifest with raw bar-table metadata. Adjusted or unknown bar basis is
rejected before the strategy runs. ML4T Data's
`ml4t.data.adjustments.prepare_action_inputs` validates and normalizes event facts;
the caller must supply genuinely known publication, currency and missing economic
terms. A bundle export's null publication field is not permission to invent one.

`to_actions_dataframe()` retains application times, revision identity, economic
terms, cash changes and order adjustments. `to_entitlements_dataframe()` retains
eligibility and payment facts even after the source position is sold. Account
observations include actual `receivables_value`. Persist these outputs with
`result.to_parquet(...)`; action-enabled results use artifact schema 5 with
required action and entitlement tables and content verification. Older artifacts
remain readable, with missing action history reported as unavailable.

Adjusted account marks preserve the original quote observation timestamp.
`broker.get_last_price()` and runtime history keep the raw observations. An
account basis adjustment is not an executable new quote. Security delivery requires
an observed destination mark; it does not create a market observation.

## Zipline migration differences and tested scope

Set `BacktestConfig(corporate_action_policy="zipline_3_2")` for the tested legacy
rounding and recognition method. The callback `run_algorithm` adapter selects this
method. It truncates fractional shares toward zero, uses rounded reported basis
for split cash-in-lieu, and recognizes distributions at payment. Native
`declared_terms` recognizes unpaid cash/security receivables when earned.

The retained original is published `zipline-reloaded==3.2.0`, source
`cf649ac06fddce76f6caa2403cd8494ac84f72ad`, wheel SHA-256
`1d7a3c4f547c02e9a3c3cc1f8ea2394641124c1fdb35ef27cdf30fa488d3899b`.
Actual snapshots were exported and native action facts prepared by ML4T Data
source `c3b9db80a66397420fb27adeccc762559dd41e07`
(`0.2.2.dev21+gc3b9db80a`) with published Specs 0.2.0. These development Data
artifacts are qualification identities, not a claim that an older published Data
release contains these interfaces.

`tests/test_zipline_corporate_actions.py` consumes those exported inputs and compares
closing cash, NAV and positions for long/short splits, reverse splits, partial orders,
cash payments after sale, own/cross-stock payments and destination position changes.
The original reports transaction commission as `None` under `NoCommission`; native
fills report explicit zero, while original and native order commission are both zero.
After a security payment covers a destination position, the callback facade can
retain the original's zero-quantity position entry and reported basis. Native
holdings remain flat; the facade projects that presentation from the payment journal.

Original daily premarket valuation can already read that day's close. Changing only
the ex-day close by $7 changes its premarket mark by $7. Backtest retains the causal
previous adjusted mark until the quote arrives. Those premarket values deliberately
differ; closing ledger agreement does not establish premarket information parity.
Native mixed cash/security mergers use independent accounting oracles. The retained
original comparisons do not establish general merger, tax-lot or multi-currency parity.

Continue with [Zipline migration](migrate-from-zipline.md) for callbacks and prepared
inputs, or [Results and Analysis](results.md) for account exports.
