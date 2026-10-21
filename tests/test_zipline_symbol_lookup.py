"""Exported ownership policy must not become future market information or listing eligibility."""

import io
import json
import math
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm

_ROOT = Path(__file__).parent / "fixtures/zipline_symbol_lookup"
_REFERENCE = json.loads((_ROOT / "reference.json").read_text())


def _snapshot(actual=False):
    with tarfile.open(_ROOT / "snapshot.tar.gz") as archive:
        if actual:
            storage = pytest.importorskip("ml4t.data.storage")
            with TemporaryDirectory(prefix="ml4t-symbol-snapshot-") as directory:
                archive.extractall(directory, filter="data")
                return storage.read_snapshot(directory)
        manifest = json.loads(archive.extractfile("manifest.json").read())
        tables = {
            name: pl.read_parquet(io.BytesIO(archive.extractfile(entry["files"][0]["path"]).read()))
            for name, entry in manifest["content"]["tables"].items()
        }
    return SimpleNamespace(snapshot_id=manifest["snapshot_id"], manifest=manifest, tables=tables)


def _feed(snapshot):
    clock = snapshot.manifest["content"]["tables"]["minute_bars"]["clock"]
    spec = FeedSpec.from_any(clock["feed_spec"]).with_overrides(calendar=clock["resolved_calendar"])
    return DataFeed(prices_df=snapshot.tables["minute_bars"], feed_spec=spec)


@pytest.mark.parametrize(
    "actual", [False, True], ids=["captured-data-output", "installed-data-reader"]
)
def test_exported_symbol_policy_matches_end_session_date_override_country_and_share_class(actual):
    snapshot = _snapshot(actual)
    from ml4t.backtest.compat.zipline import PreparedAssetLookup

    lookup = PreparedAssetLookup(snapshot)
    initial, trace = [], []

    def describe(asset):
        return {"sid": int(asset), "symbol": asset.symbol}

    def initialize(context):
        context.new = context.symbol("ALPHA")
        context.n = 0
        initial.append({"default": describe(context.new)})
        context.set_symbol_lookup_date("2025-07-02")
        initial[0]["old"] = describe(context.symbol("ALPHA"))
        result = context.symbols("brk.b", "BRK/B", country_code="US")
        initial[0]["symbols_type"] = type(result).__name__
        initial[0]["symbols"] = [describe(asset) for asset in result]
        context.set_symbol_lookup_date("2025-07-02T20:00:00-04:00")
        initial[0]["utc_override"] = describe(context.symbol("ALPHA"))
        context.set_symbol_lookup_date("2025-07-05")
        initial[0]["after_lifetime"] = describe(context.symbol("BETA"))
        context.set_symbol_lookup_date("2025-07-03")

    def handle(context, data):
        if context.n in [0, 389, 390, 599]:
            value = data.current(context.new, "price")
            trace.append(
                {
                    "timestamp": context.get_datetime().isoformat(),
                    "sid": int(context.symbol("ALPHA")),
                    "new_price": None if math.isnan(value) else value,
                }
            )
        context.n += 1

    frame = run_algorithm(
        feed=_feed(snapshot),
        asset_lookup=lookup,
        initialize=initialize,
        handle_data=handle,
        start="2025-07-02",
        end="2025-07-03",
        capital_base=1000,
    )
    assert initial == _REFERENCE["initial"]
    assert trace == _REFERENCE["trace"]
    assert len(frame) == _REFERENCE["callback_count"]
    assert (
        frame.attrs["backtest_result"].config.metadata["migration_symbol_lookup"]["source_id"]
        == snapshot.snapshot_id
    )


def test_lookup_keeps_cloned_facts_and_readonly_provenance_when_caller_mutates_snapshot():
    snapshot = _snapshot()
    from ml4t.backtest.compat.zipline import PreparedAssetLookup

    lookup = PreparedAssetLookup(snapshot)
    snapshot.tables["zipline_symbol_lookup"] = snapshot.tables[
        "zipline_symbol_lookup"
    ].with_columns(pl.lit("WRONG").alias("company_symbol"))
    assert int(lookup.resolve("ALPHA", "2025-07-02")) == 0
    assert int(lookup.resolve("ALPHA", "2025-07-03")) == 1
    assert int(lookup.resolve("BETA", "2025-07-05")) == 0
    metadata = lookup.metadata
    metadata["source_id"] = "changed"
    assert lookup.metadata["source_id"] == snapshot.snapshot_id


def test_missing_or_ambiguous_owned_symbol_is_not_silently_replaced_by_static_display_name():
    snapshot = _snapshot()
    from ml4t.backtest.compat.zipline import PreparedAssetLookup

    lookup = PreparedAssetLookup(snapshot)
    with pytest.raises(ValueError, match="symbol"):
        lookup.resolve("UNKNOWN", "2025-07-03")
    with pytest.raises(ValueError, match="symbol"):
        lookup.resolve("ALPHA", "2025-07-03", country_code="XX")
    with pytest.raises(ValueError, match="snapshot|date|known"):
        lookup.resolve("ALPHA", "NaT")


def test_static_symbol_calls_return_the_published_list_container_and_share_class_spellings():
    from ml4t.backtest.compat.zipline.api import symbols
    from ml4t.backtest.compat.zipline.assets import Asset

    observed = []

    def initialize(context):
        observed.append(symbols("ALPHA", "ALPHA"))
        assert context.symbol("brk.b") == context.sid(2)
        assert context.symbol("BRK/B") == context.sid(2)

    run_algorithm(
        feed=_feed(_snapshot()),
        assets=[Asset("symbol-demo:1", "ALPHA", 1), Asset("symbol-demo:2", "BRK_B", 2)],
        initialize=initialize,
    )
    assert isinstance(observed[0], list)
    assert [int(asset) for asset in observed[0]] == [1, 1]


def test_date_and_country_operations_require_facts_instead_of_accepting_and_ignoring_them():
    from ml4t.backtest.compat.zipline.assets import Asset

    def initialize(context):
        with pytest.raises(ValueError, match="PreparedAssetLookup"):
            context.set_symbol_lookup_date("2025-07-02")
        with pytest.raises(ValueError, match="PreparedAssetLookup"):
            context.symbol("ALPHA", country_code="US")

    run_algorithm(
        feed=_feed(_snapshot()), assets=[Asset("symbol-demo:1", "ALPHA", 1)], initialize=initialize
    )


def test_calendar_none_names_observation_date_policy_without_inventing_an_end_session():
    from ml4t.backtest.compat.zipline import PreparedAssetLookup

    snapshot = _snapshot()
    original = _feed(snapshot)
    feed = DataFeed(
        prices_df=original.prices, feed_spec=original.feed_spec.with_overrides(calendar=None)
    )
    result = run_algorithm(feed=feed, asset_lookup=PreparedAssetLookup(snapshot))
    metadata = result.attrs["backtest_result"].config.metadata["migration_symbol_lookup"]
    assert metadata["default_policy"] == "end_observation_UTC_date"
    assert metadata["default_date"] == "2025-07-03T00:00:00"


def test_exported_lookup_drives_actual_fill_and_reloaded_account_with_published_trade_trace(
    tmp_path,
):
    from ml4t.backtest import BacktestResult
    from ml4t.backtest.compat.zipline import PreparedAssetLookup, to_performance_frame
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    reference = json.loads((_ROOT / "trading-reference.json").read_text())
    snapshot = _snapshot()
    trace = []

    def initialize(context):
        context.new = context.symbol("ALPHA")
        context.n = 0
        context.set_commission(commission.PerShare(0, 0))
        context.set_slippage(slippage.FixedSlippage(0))

    def handle(context, data):
        if context.n == 390:
            context.order(context.new, 1)
        if context.n in [0, 389, 390, 391, 599]:
            price = data.current(context.new, "price")
            trace.append(
                {
                    "timestamp": context.get_datetime().isoformat(),
                    "sid": int(context.symbol("ALPHA")),
                    "new_price": None if math.isnan(price) else price,
                    "cash": context.portfolio.cash,
                    "equity": context.portfolio.portfolio_value,
                    "quantity": context.portfolio.positions[context.new].amount,
                }
            )
        context.n += 1

    frame = run_algorithm(
        feed=_feed(snapshot),
        asset_lookup=PreparedAssetLookup(snapshot),
        initialize=initialize,
        handle_data=handle,
        start="2025-07-02",
        end="2025-07-03",
        capital_base=1000,
    )
    assert trace == reference["trace"]
    actual = [
        {
            "timestamp": item["dt"].isoformat(),
            "sid": int(item["sid"]),
            "amount": item["amount"],
            "price": item["price"],
        }
        for group in frame.transactions
        for item in group
    ]
    assert actual == reference["fills"]
    native = frame.attrs["backtest_result"]
    assert native.fills[0].asset == "symbol-demo:1"
    native.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    reopened = to_performance_frame(restored)
    assert reopened.cash.iloc[-1] == 950 and reopened.portfolio_value.iloc[-1] == 1000
    transaction = next(group[0] for group in reopened.transactions if group)
    assert int(transaction["sid"]) == 1 and transaction["sid"].key == "symbol-demo:1"
    assert transaction["sid"].symbol == "ALPHA" and transaction["commission"] == 0
    assert restored.config.metadata["migration_symbol_lookup"]["source_id"] == snapshot.snapshot_id
