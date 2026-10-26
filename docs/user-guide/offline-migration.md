# Reproduce an offline research-to-analysis run

Prepare data and features before simulation, execute through the existing
`Strategy` and `Engine`, then send the saved result to Diagnostic. The runnable
[`offline_migration.py`](https://github.com/ml4t/backtest/blob/45becfb0985eb59e63c7e3d3f513dba372b103da/examples/offline_migration.py)
example demonstrates this route with synthetic equities. It needs no Zipline,
Pyfolio, data-provider credentials or network connection during execution.

Backtest 0.2.0 contains these interfaces. The complete analysis route also
requires Diagnostic 0.2.0; the published Diagnostic 0.1.9 wheel does not contain
its newer holdings and legacy-result helpers. Keep the package and source
installation routes separate.

## Versioned package installation

Use Backtest 0.2.0, Data 0.3.1, Engineer 0.2.1, Diagnostic 0.2.0 and Specs 0.2.0
on Python 3.12-3.14. Run this route only after
all five versions are available on PyPI. Until then, use the qualified source
cohort below rather than mixing a newer script with an older installed API.

From a checkout containing the linked example, install the exact cohort and
run the eager workflow:

```bash
uv run --no-project --python 3.12 \
  --with 'ml4t-backtest==0.2.0' --with 'ml4t-data==0.3.1' \
  --with 'ml4t-engineer[calendars]==0.2.1' --with 'ml4t-diagnostic==0.2.0' \
  --with 'ml4t-specs==0.2.0' --with 'pandas>=3.1' --with 'pyarrow>=26' \
  python -I examples/offline_migration.py ./offline-run
```

Append `--lazy` and choose a different output directory for the lazy-input
variant. The expected output and numerical checks below apply to both routes.
Keep these package versions together; the preserved source builds below have
different version labels and do not establish compatibility with older PyPI wheels.

## Preserved qualified source cohort

| Component | Tested identity |
|---|---|
| Backtest | Original source-cohort example [e811285c](https://github.com/ml4t/backtest/tree/e811285c99d230602bb8e5bac39c86f0979b0254); the current released package is 0.2.0. |
| Data | `0.2.2.dev21+gc3b9db80a`, [source c3b9db80](https://github.com/ml4t/data/tree/c3b9db80a66397420fb27adeccc762559dd41e07). |
| Engineer | `0.1.7.dev58+g093b72d68`, [source 093b72d6](https://github.com/ml4t/engineer/tree/093b72d6801cf2b279f1aa682c6d2cabe45d494d). |
| Diagnostic | Distribution label `0.1.9`, [source 519aab16](https://github.com/ml4t/diagnostic/tree/519aab16eb11567c33072b4befa76aef4ecc627a). This identifies the tested source build; it does not claim the published 0.1.9 wheel contains its newer holdings and legacy-result helpers. |
| Specs | Published `0.2.0`. |

Use a separate analysis environment with the built Backtest wheel and these
producer/consumer sources. The
[installed integration workflow](https://github.com/ml4t/backtest/blob/main/.github/workflows/successor-integration.yml)
contains the exact installation commands. It tests Python 3.12, 3.13 and 3.14
on Linux, macOS and Windows with isolated wheel imports and no skipped workflows.
The suite includes causal history, Pipeline, identity, lifecycle, controls and
accounting paths using the real producers. Qualification uses Pandas 3.1 or later
to exercise current calendar dependency compatibility.
The Arrow-present cohort uses PyArrow 26 or later with Pandas 3.1 so legacy
timestamp conversion avoids the deprecated timezone-dropping index accessor.
Use that interpreter range for this pinned cohort: its Data and Diagnostic builds
declare Python below 3.15. Backtest core Python 3.15 checks run separately.
Core Backtest remains usable without the optional Data, Engineer and Diagnostic packages.

## Execute the example

From the source checkout, with the qualified packages installed, run:

```bash
python -I examples/offline_migration.py ./offline-run
```

The lazy feature-input variant uses the same workflow:

```bash
python -I examples/offline_migration.py ./offline-run-lazy --lazy
```

Both commands print:

```text
observations=5 fills=1 total_return=0.000800
```

Each output directory contains:

- `raw-inputs`: Data's immutable prices, universe facts and decision declarations.
- `prepared-inputs`: prices and the actual Engineer feature artifact, with original
  source snapshot identity, universe and decision inputs.
- `result`: actual native account, holdings, fills, configuration and result manifest.
- `workflow.json`: intermediate features, delivered callback values, consumed
  manifest identity and actual Diagnostic holdings/activity outputs.

The generated input snapshots are immutable. Repeating the same preparation can
reuse them; different source data or declarations require a new output directory.
The example's `random_seed=None` records that it performs no random sampling.
Its strategy definition and explicit execution settings are retained with the data.

## Understand the checks

Data prepares raw history separately at each physical decision cutoff. Engineer
calculates a three-observation mean, using already-known universe facts. Asset A's
means are 101 and 102. Asset B's raw means are 199 and 198, but its declared
membership is false: the Backtest callbacks receive no usable B feature and
submit no B order. Neither the raw rows nor their identities are silently dropped.

The first eligible A decision occurs on February 5 at 21:00 UTC. Ten shares fill
at the next observed close, 103, on February 6, with a $2 commission. Cash becomes
$8,968 and NAV $9,998. The final observed price, 104, makes NAV $10,008. The
position remains open. Diagnostic reads all five observations, including the
initial all-cash periods, and reports a total investment return of 0.08%.
Actual transaction analysis retains $1,030 of traded notional and the $2 fee.

This is an observation-clock example with `calendar=None` and an explicitly
chosen annualization of 252 observations. It does not infer US intraday hours.
The recipe supplies no benchmark, factor-return or market-volume dataset; its
verified outputs are returns, observed holdings and actual trading activity.
It makes no benchmark-relative, factor-attribution or market-participation claim.

## Consume a verified snapshot in your strategy

Use `DataFeed.from_snapshot(path, prices_table="prices", signals_table="features")`.
Data's public reader verifies the manifest, every declared file hash and schema
before Backtest selects explicitly named tables. Specify optional `context_table`,
`corporate_actions_table` and `capital_flows_table` when those inputs are present.
Unknown table names fail. Extra captured tables remain in source provenance;
the result identifies exactly which tables Backtest consumed.

The feed inherits the source's effective prepared calendar, including explicit
`None`. An explicit `calendar` argument overrides it. The declared price FeedSpec
selects columns; the receiver preserves the captured stable IDs. It does not guess
symbols, FX conversion, missing lifecycle terms or observation instants for dates.
Prepare session labels as physical instants when using native runtime services.

When the selected signal table carries an Engineer feature artifact, Backtest
uses its declared decision clock, publication columns, eligibility and feature
columns. Data may rewrite the artifact's storage location when capturing it;
Backtest retains that actual captured declaration. An explicit `feature_spec`
can select a different declared projection; its effective specification and
projection are also recorded. It does not execute an arbitrary Pipeline object.
See [prepared Pipeline migration](migrate-from-zipline.md#deliver-prepared-pipeline-results).

Pandas, indexed Pandas and eager/lazy Polars prices/context normalize at the feed
boundary. Named index keys, nanosecond clocks, nullable values and equal-time
asset order are preserved. Feature panels continue to require their explicit
feature specification. Existing `prices_path`, `signals_path` and `context_path`
accept saved Parquet inputs; use
[prepared CSV delivery](migrate-from-zipline.md#register-an-offline-csv-source)
for the legacy CSV research route. Ambiguous duplicate or null event keys fail;
economically distinct events cannot silently overwrite one another.

Treat snapshot feeds as immutable. Backtest compares normalized table identities
before callbacks and after execution. An in-memory price change after loading a
snapshot cannot retain its original source attestation. Prepare a new snapshot
for a different universe, price transformation or data revision.

`result.input_provenance` retains the full source manifest, snapshot identity,
source manifest checksum, exact manifest-file hash, selected tables and normalized
table identities. The result archive binds this receipt to its content digest.
The saved native configuration records the effective execution settings. A direct
caller-supplied frame without a verified snapshot has no snapshot attestation.
Data owns input storage/adjustment policy, Engineer owns feature construction,
Backtest owns execution/account observations, and Diagnostic owns analytics.

## Preserve material legacy differences

The existing [Zipline migration guide](migrate-from-zipline.md) covers callback,
history, order/cost/control and prepared Pipeline/CSV routes. The original dual
moving-average and OLMAR ports retain source attribution and separate callback,
history, intent, fill and account checks. Source preparation replaces the old
bundle dependency; no second execution engine is introduced.

Diagnostic's `portfolio_analysis_from_zipline` separately consumes captured original
Zipline 3.2.0 performance frames. The test fixture retains original reported
returns, cash, NAV, complete position lists and actual transactions. Asset objects
are transported as integer SIDs with explicit multiplier metadata. Legacy signed
`amount` remains signed in its execution quantity; native exports provide separate
quantity, side and signed quantity fields. Unknown transaction fees/slippage stay
unknown rather than being inferred from portfolio costs.

For futures, keep economic notional separate from cash and liquidation value.
The tested native variation account pays $500 for a one-contract 4000-to-4010 move
with multiplier 50; it does not move $200,000 of principal through cash. Diagnostic
retains that exposure alongside authoritative NAV and cash. Follow the qualified
[variation accounting](futures-accounting.md),
[dated rolls/lifecycle](dated-futures.md) and
[raw corporate-action](corporate-actions.md) recipes for those methods. Different
exchange settlement, adjusted-history or synthetic expiry assumptions need their
own explicit input contract and comparison.
