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


def test_snapshot_feature_artifact_drives_actual_engineer_decision_clock_and_eligibility(tmp_path):
    from pathlib import Path

    storage = pytest.importorskip("ml4t.data.storage")
    preparation = pytest.importorskip("ml4t.engineer.features.preparation")
    artifacts = pytest.importorskip("ml4t.engineer.artifacts")
    rolling = pytest.importorskip("ml4t.engineer.features.rolling")
    root = Path(__file__).parent / "fixtures/prepared_features"
    source = json.loads((root / "manifest.json").read_text())
    panel = preparation.prepare_feature_panel(
        pl.read_parquet(root / "history.parquet"),
        features={"average": rolling.finite_mean("close", period=3)},
        lookbacks={"average": 3},
        input_columns=["close"],
        decisions=pl.read_parquet(root / "decisions.parquet"),
        universe=pl.read_parquet(root / "universe.parquet"),
        price_basis="raw",
        feed_spec=FeedSpec.from_any(source["input_feed_spec"]),
    )
    feature_spec = artifacts.FeatureSpec.from_mapping(source["feature_spec"])
    prices = panel.select(
        pl.col("decision_time").alias("timestamp"), "asset", pl.lit(100.0).alias("close")
    )
    metadata = {
        "source_revision": "declared-fixture-r1",
        "identity_namespace": "fixture",
        "units": {"close": "USD/share", "average": "USD/share"},
        "as_of": None,
        "price_basis": "raw",
        "share_basis": "source",
        "revisions": {"fixture": "r1"},
    }
    storage.export_snapshot(
        tmp_path / "inputs",
        {
            "prices": storage.SnapshotTable(
                prices,
                role="prices",
                metadata=metadata,
                feed_spec=FeedSpec(
                    time_contract_version=2, calendar=None, timestamp_semantics="event_time"
                ),
            ),
            "features": storage.SnapshotTable(
                panel, role="features", metadata=metadata, artifact=feature_spec
            ),
        },
        effective_config={"seed": 7},
        artifacts={},
        calendar=None,
    )
    feed = DataFeed.from_snapshot(tmp_path / "inputs", signals_table="features")
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append([data[asset]["signals"]["average"] for asset in ["A", "B"]])
            if data["A"]["signals"]["average"] is not None and broker.get_position("A") is None:
                broker.submit_order("A", 1)

    result = Engine(
        feed,
        Read(),
        BacktestConfig(
            calendar=None,
            initial_cash=10000,
            execution_mode=ExecutionMode.SAME_BAR,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        holdings_sampling="bar",
    ).run()
    assert seen == [[2.0, None], [3.0, None]]
    assert result.to_fills_dataframe()["asset"].to_list() == ["A"]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [9898, 9898]
    assert result.input_provenance["feature_spec"]["schema"] == feature_spec.to_dict()["schema"]
    assert (
        result.input_provenance["feature_spec"]["definition"]
        == feature_spec.to_dict()["definition"]
    )
    assert result.input_provenance["feature_spec"]["storage"]["path"] == "tables/features"
    result.to_parquet(tmp_path / "results")
    assert (
        BacktestResult.from_parquet(tmp_path / "results").input_provenance
        == result.input_provenance
    )


def test_real_snapshot_consumes_all_declared_tables_without_losing_actions_or_cash_flows(tmp_path):
    storage = pytest.importorskip("ml4t.data.storage")
    from ml4t.backtest import ContractSpec

    SnapshotTable, export_snapshot = storage.SnapshotTable, storage.export_snapshot
    times = [datetime(2025, 1, 6, 21, tzinfo=UTC) + timedelta(days=i) for i in range(3)]
    prices = pl.DataFrame({"timestamp": times, "asset": ["A"] * 3, "close": [100.0, 50.0, 52.0]})
    signals = pl.DataFrame({"timestamp": times, "asset": ["A"] * 3, "score": [10.0, 20.0, 30.0]})
    context = pl.DataFrame({"timestamp": times, "risk": [1.0, 2.0, 3.0]})
    flows = pl.DataFrame(
        {"timestamp": [times[1]], "event_id": ["fund"], "kind": ["delta"], "value": [1000.0]}
    )
    actions = pl.DataFrame(
        [
            {
                "event_id": "split",
                "action_kind": "split",
                "asset": "A",
                "available_at": times[0],
                "effective_at": times[1],
                "currency": "USD",
                "source": "capture",
                "revision_id": "r1",
                "share_basis": "pre_event",
                "split_ratio": 2.0,
                "split_ratio_convention": "new_per_old",
            }
        ]
    )
    frames = {
        "prices": prices,
        "signals": signals,
        "context": context,
        "corporate_actions": actions,
        "capital_flows": flows,
    }
    metadata = {
        "source_revision": "all-roles-r1",
        "identity_namespace": "capture",
        "units": {"close": "USD/share"},
        "as_of": times[-1].isoformat(),
        "price_basis": "raw",
        "share_basis": "source",
        "revisions": {"capture": "r1"},
    }
    spec = FeedSpec(
        time_contract_version=2, calendar=None, timestamp_semantics="event_time", entity_col="asset"
    )
    snapshot = export_snapshot(
        tmp_path / "inputs",
        {
            name: SnapshotTable(
                frame,
                role=name,
                metadata=metadata,
                feed_spec=spec if name in {"prices", "signals"} else None,
            )
            for name, frame in frames.items()
        },
        effective_config={},
        artifacts={},
        calendar=None,
    )
    feed = DataFeed.from_snapshot(
        tmp_path / "inputs",
        signals_table="signals",
        context_table="context",
        corporate_actions_table="corporate_actions",
        capital_flows_table="capital_flows",
    )
    observed = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            observed.append((data["A"]["signals"]["score"], context["risk"]))
            if timestamp == times[0]:
                broker.submit_order("A", 10)

    result = Engine(
        feed,
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=10000,
            commission_rate=0,
            slippage_rate=0,
            execution_mode=ExecutionMode.SAME_BAR,
        ),
        contract_specs={"A": ContractSpec("A", currency="USD")},
        holdings_sampling="bar",
    ).run()
    assert observed == [(10, 1), (20, 2), (30, 3)]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [9000, 10000, 10000]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000, 11000, 11040]
    assert result.to_actions_dataframe().filter(pl.col("operation") == "split")[
        "quantity_after"
    ].to_list() == [20]
    assert result.input_provenance["snapshot_id"] == snapshot.snapshot_id
    assert result.input_provenance["consumed_tables"] == {name: name for name in frames}
    result.to_parquet(tmp_path / "result")
    assert (
        BacktestResult.from_parquet(tmp_path / "result").input_provenance == result.input_provenance
    )
