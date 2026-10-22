"""Real immutable Data inputs must be verified and identified in native result artifacts."""

import hashlib
import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.backtest import BacktestConfig, BacktestResult, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode


def capture_inputs(path, *, declared_calendar=None):
    storage = pytest.importorskip("ml4t.data.storage")
    times = [datetime(2025, 1, 6, 21, tzinfo=UTC) + timedelta(days=i) for i in range(3)]
    table = storage.SnapshotTable(
        pl.DataFrame(
            {"timestamp": times, "asset": ["capture:A"] * 3, "close": [100.0, 101.0, 102.0]}
        ),
        role="prices",
        metadata={
            "source_revision": "declared-fixture-r1",
            "identity_namespace": "capture",
            "units": {"close": "USD/share"},
            "as_of": times[-1].isoformat(),
            "price_basis": "raw",
            "share_basis": "source",
            "revisions": {"capture": "r1"},
        },
        feed_spec=FeedSpec(
            time_contract_version=2, calendar=declared_calendar, timestamp_semantics="event_time"
        ),
    )
    return storage.export_snapshot(
        path, {"prices": table}, effective_config={"seed": 7}, artifacts={}, calendar=None
    )


@pytest.mark.parametrize("declared_calendar", [None, "NYSE"])
def test_verified_snapshot_records_exact_consumed_manifest_and_preserves_effective_null_calendar(
    tmp_path, declared_calendar
):
    snapshot = capture_inputs(tmp_path / "inputs", declared_calendar=declared_calendar)
    feed = DataFeed.from_snapshot(tmp_path / "inputs")

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if broker.get_position("capture:A") is None:
                broker.submit_order("capture:A", 10)

    result = Engine(
        feed,
        Buy(),
        BacktestConfig(
            initial_cash=10000,
            execution_mode=ExecutionMode.SAME_BAR,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        holdings_sampling="bar",
    ).run()
    assert result.config.resolved_calendar is None
    assert result.to_equity_dataframe()["equity"].to_list() == [9998, 10008, 10018]
    origin = result.input_provenance
    assert origin["snapshot_id"] == snapshot.snapshot_id
    assert origin["manifest_sha256"] == snapshot.manifest["manifest_sha256"]
    assert (
        origin["manifest_file_sha256"]
        == hashlib.sha256((tmp_path / "inputs/manifest.json").read_bytes()).hexdigest()
    )
    assert origin["manifest"] == snapshot.manifest
    assert origin["consumed_tables"] == {"prices": "prices"}
    result.to_parquet(tmp_path / "results")
    restored = BacktestResult.from_parquet(tmp_path / "results")
    assert restored.input_provenance == origin
    metadata = json.loads((tmp_path / "results/manifest.json").read_text())
    metadata["input_provenance"]["consumed_tables"]["prices"] = "missing"
    (tmp_path / "results/manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="[Ii]nput provenance"):
        BacktestResult.from_parquet(tmp_path / "results")


def test_changed_snapshot_file_is_rejected_before_any_simulation_starts(tmp_path):
    capture_inputs(tmp_path / "inputs")
    table = tmp_path / "inputs/tables/prices/data.parquet"
    pl.read_parquet(table).with_columns((pl.col("close") + 1).alias("close")).write_parquet(table)
    with pytest.raises(ValueError, match="[Cc]hecksum|[Hh]ash"):
        DataFeed.from_snapshot(tmp_path / "inputs")


def test_in_memory_price_change_after_snapshot_read_is_rejected_before_start_callback(tmp_path):
    capture_inputs(tmp_path / "inputs")
    feed = DataFeed.from_snapshot(tmp_path / "inputs")
    feed.prices = feed.prices.with_columns((pl.col("close") + 1).alias("close"))
    started = []

    class Noop(Strategy):
        def on_start(self, broker):
            started.append(True)

        def on_data(self, *args):
            pass

    with pytest.raises(ValueError, match="[Ss]napshot.*changed|[Cc]hanged.*snapshot"):
        Engine(feed, Noop(), BacktestConfig(calendar=None)).run()
    assert not started
