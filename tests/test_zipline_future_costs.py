"""Original futures fee options must reach the real native partial-fill ledger."""

import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from ml4t.backtest import AssetClass, ContractSpec, DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

ROOT = Path(__file__).parent / "fixtures/zipline_future_costs"
REFERENCE = json.loads((ROOT / "reference.json").read_text())


def _feed():
    return DataFeed(
        prices_df=pl.DataFrame(
            [
                {
                    "timestamp": datetime(2025, 3, day, 21, tzinfo=UTC),
                    "asset": asset,
                    "close": 100.0,
                    "volume": 2.0,
                }
                for day in [3, 4, 5, 6]
                for asset in ["A", "ES"]
            ]
        ),
        feed_spec={
            "calendar": None,
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
        },
    )


@pytest.mark.parametrize("method", ["per_contract", "per_future_trade"])
@pytest.mark.parametrize("mapping", [False, True], ids=["scalar", "root_mapping"])
def test_original_future_fee_parameters_preserve_partial_order_minimum_exchange_fee_and_multiplier(
    method, mapping
):
    from ml4t.backtest.compat.zipline.api import future_symbol, set_commission
    from ml4t.backtest.compat.zipline.assets import Future
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    future = Future("ES", "ESH25", 8, root_symbol="ES", multiplier=50.0, tick_size=0.25)

    def value(number):
        return {"ES": number} if mapping else number

    def initialize(context):
        model = (
            commission.PerContract(value(2.0), value(3.0), 8.0)
            if method == "per_contract"
            else commission.PerFutureTrade(value(3.0))
        )
        set_commission(us_equities=commission.PerShare(0.2, 1.0), us_futures=model)
        context.set_slippage(
            us_equities=slippage.VolumeShareSlippage(volume_limit=0.5, price_impact=0),
            us_futures=slippage.FixedBasisPointsSlippage(0, 0.5),
        )
        context.future = future_symbol("esh25")
        assert context.future is future and context.sid(8) is future
        context.sent = False

    def handle(context, data):
        if not context.sent:
            context.order(context.sid(7), 3)
            context.order(context.future, 3)
            context.sent = True

    frame = run_algorithm(
        feed=_feed(), assets=[Asset("A", "A", 7), future], initialize=initialize, handle_data=handle
    )
    result = frame.attrs["backtest_result"]
    reference = json.loads(
        (Path(__file__).parent / "fixtures/zipline_original/future-fee-reference.json").read_text()
    )["methods"][method]
    assert result.to_equity_dataframe()["equity"].to_list() == reference["nav"]
    assert result.to_fills_dataframe().filter(pl.col("asset") == "ES")["commission"].to_list() == (
        [8.0, 0.0, 1.0] if method == "per_contract" else [3.0, 0.0, 0.0]
    )
    assert result.to_fills_dataframe()["quantity"].to_list() == [1.0] * 6
    assert result.to_portfolio_state_dataframe()["cash"][-1] == (
        84690.0 if method == "per_contract" else 84696.0
    )


def test_future_contract_multiplier_is_used_for_value_orders_instead_of_equity_quantity():
    from ml4t.backtest.compat.zipline.assets import Future
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    future = Future("ES", "ESH25", 8, root_symbol="ES", multiplier=50.0, tick_size=0.25)

    def initialize(context):
        context.sent = False
        context.set_commission(us_futures=commission.PerFutureTrade(0))
        context.set_slippage(us_futures=slippage.NoSlippage())

    def handle(context, data):
        if not context.sent:
            context.order_value(future, 5000)
            context.sent = True

    result = run_algorithm(
        feed=_feed(), assets=[future], initialize=initialize, handle_data=handle
    ).attrs["backtest_result"]
    assert result.to_fills_dataframe()["quantity"].to_list() == [1.0]
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 95000.0


def test_conflicting_declared_future_spec_fails_before_initialize_or_account_mutation():
    from ml4t.backtest.compat.zipline.assets import Future

    seen = []
    future = Future("ES", "ESH25", 8, root_symbol="ES", multiplier=50.0, tick_size=0.25)
    with pytest.raises(ValueError, match="conflicts"):
        run_algorithm(
            feed=_feed(),
            assets=[future],
            initialize=lambda c: seen.append(c),
            contract_specs={"ES": ContractSpec("ES", AssetClass.FUTURE, multiplier=20)},
        )
    assert seen == []


def _snapshot(actual):
    with tarfile.open(ROOT / "snapshot.tar.gz") as archive:
        if actual:
            storage = pytest.importorskip("ml4t.data.storage")
            with TemporaryDirectory(prefix="ml4t-future-facts-") as directory:
                archive.extractall(directory, filter="data")
                return storage.read_snapshot(directory)
        manifest = json.loads(archive.extractfile("./manifest.json").read())
        tables = {
            name: pl.read_parquet(
                io.BytesIO(archive.extractfile("./" + entry["files"][0]["path"]).read())
            )
            for name, entry in manifest["content"]["tables"].items()
        }
        return SimpleNamespace(snapshot_id=manifest["snapshot_id"], tables=tables)


@pytest.mark.parametrize("actual", [False, True], ids=["captured_data", "installed_data_reader"])
@pytest.mark.parametrize(
    "method", ["cost_fallback", "exchange_fallback", "default", "trade_fallback"]
)
def test_actual_exported_future_facts_and_reference_fee_fallbacks_drive_account_and_saved_result(
    actual, method, tmp_path
):
    from ml4t.backtest.compat.zipline import PreparedAssetLookup, to_performance_frame
    from ml4t.backtest.compat.zipline.assets import Future
    from ml4t.backtest.compat.zipline.finance import commission, slippage
    from ml4t.backtest.result import BacktestResult

    snapshot = _snapshot(actual)
    lookup = PreparedAssetLookup(snapshot)
    assert snapshot.snapshot_id == REFERENCE["snapshot_id"]
    raw = snapshot.tables["daily_bars"]
    labels = raw["timestamp"].unique().sort()
    # The export declares session labels. These captured original closes are physical instants.
    clocks = pl.DataFrame(
        {
            "label": labels,
            "timestamp": [datetime.fromisoformat(value) for value in REFERENCE["closes"]],
        }
    )
    feed = DataFeed(
        prices_df=raw.rename({"timestamp": "label"}).join(clocks, on="label"),
        feed_spec={
            "calendar": None,
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
        },
    )

    def initialize(context):
        context.sent = False
        context.set_commission(us_equities=commission.PerShare(0.2, 1.0))
        if method != "default":
            model = {
                "cost_fallback": commission.PerContract({}, {"ES": 3.0}),
                "exchange_fallback": commission.PerContract({"ES": 2.0}, {}),
                "trade_fallback": commission.PerFutureTrade({}),
            }[method]
            context.set_commission(us_futures=model)
        context.set_slippage(
            us_equities=slippage.VolumeShareSlippage(volume_limit=0.5, price_impact=0),
            us_futures=slippage.FixedBasisPointsSlippage(0, 0.5),
        )
        contract = context.future_symbol("esh25")
        assert isinstance(contract, Future)
        assert (contract.root_symbol, contract.multiplier, contract.tick_size) == ("ES", 50.0, 0.25)
        with pytest.raises(ValueError, match="[Ss]ymbol"):
            context.symbol("ESH25")
        with pytest.raises(ValueError, match="Future"):
            context.future_symbol("A")

    def handle(context, data):
        if not context.sent:
            context.order(context.sid(7), 3)
            context.order(context.future_symbol("ESH25"), 3)
            context.sent = True

    frame = run_algorithm(feed=feed, asset_lookup=lookup, initialize=initialize, handle_data=handle)
    native = frame.attrs["backtest_result"]
    reference = REFERENCE["methods"][method]
    assert native.to_equity_dataframe()["equity"].to_list() == pytest.approx(reference["nav"])
    cumulative = [point["orders"][1]["commission"] for point in reference["trace"]]
    assert native.to_fills_dataframe().filter(pl.col("asset") == "future-cost-options:8")[
        "commission"
    ].to_list() == pytest.approx([b - a for a, b in zip(cumulative, cumulative[1:])])
    native.to_parquet(tmp_path)
    restored = to_performance_frame(BacktestResult.from_parquet(tmp_path))
    pd.testing.assert_frame_equal(frame, restored)
    restored_future = restored["transactions"].iloc[1][1]["sid"]
    assert isinstance(restored_future, Future)
    assert (restored_future.root_symbol, restored_future.multiplier, restored_future.tick_size) == (
        "ES",
        50.0,
        0.25,
    )


def test_futures_only_snapshot_needs_no_equity_ownership_intervals():
    from ml4t.backtest.compat.zipline import PreparedAssetLookup
    from ml4t.backtest.compat.zipline.assets import Future

    snapshot = _snapshot(False)
    snapshot.tables.pop("zipline_equities")
    snapshot.tables.pop("zipline_symbol_lookup")
    assets = PreparedAssetLookup(snapshot).assets
    assert len(assets) == 1 and isinstance(assets[0], Future) and assets[0].root_symbol == "ES"


def test_unknown_root_requires_explicit_fee_and_does_not_fabricate_exchange_facts():
    from ml4t.backtest.compat.zipline.assets import Future
    from ml4t.backtest.compat.zipline.finance import commission

    calls = []
    asset = Future("ES", "UNKNOWNH25", root_symbol="UNKNOWN", multiplier=50.0, tick_size=0.25)
    with pytest.raises(ValueError, match="unknown legacy root"):
        run_algorithm(
            feed=_feed(),
            assets=[asset],
            initialize=lambda c: c.set_commission(us_futures=commission.PerContract(1.0, {})),
            handle_data=lambda c, _d: calls.append(c),
        )
    assert calls == []


def test_none_minimum_keeps_only_contract_and_one_time_exchange_fees():
    from ml4t.backtest.compat.zipline.assets import Future
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    asset = Future("ES", "ESH25", root_symbol="ES", multiplier=50.0, tick_size=0.25)

    def initialize(context):
        context.sent = False
        context.set_commission(us_futures=commission.PerContract(2.0, 3.0, None))
        context.set_slippage(us_futures=slippage.FixedBasisPointsSlippage(0, 0.5))

    def handle(context, data):
        if not context.sent:
            context.order(asset, 3)
            context.sent = True

    result = run_algorithm(
        feed=_feed(), assets=[asset], initialize=initialize, handle_data=handle
    ).attrs["backtest_result"]
    assert result.to_fills_dataframe()["commission"].to_list() == [5.0, 2.0, 2.0]
