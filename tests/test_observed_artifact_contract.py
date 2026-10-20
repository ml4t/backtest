"""Reject misleading observed-account metadata through real Engine artifact workflows."""

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode
from ml4t.backtest.result import (
    ArtifactError,
    ArtifactManifestError,
    ArtifactReadError,
    BacktestResult,
)


@pytest.fixture
def observed_result():
    class HoldRecord(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if not broker.positions:
                broker.submit_order("TEST", 10)
            broker.record(signal=float("nan"))

    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, d, 16, tzinfo=UTC) for d in [2, 3, 6]],
            "asset": ["TEST"] * 3,
            "close": [100.0] * 3,
        }
    )
    return Engine(
        DataFeed(prices_df=prices),
        HoldRecord(),
        BacktestConfig(
            initial_cash=10000,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=10,
        ),
        holdings_sampling="bar",
    ).run()


@pytest.mark.parametrize(
    "fault", ["duplicate_chunks", "wrong_bar_ids", "missing_name", "untagged_nan"]
)
def test_reader_rejects_record_writer_faults_without_exposing_ambiguous_signals(
    observed_result, tmp_path, monkeypatch, fault
):
    original = observed_result.to_recorded_dataframe

    def faulty_export():
        frame = original()
        if fault == "duplicate_chunks":
            return pl.concat([frame, frame])
        if fault == "wrong_bar_ids":
            # Confusing account observation IDs with return-bar IDs shifts the foreign key.
            return frame.with_columns(
                (pl.col("bar_index") + len(observed_result.equity_curve)).alias("bar_index")
            )
        if fault == "missing_name":
            return frame.drop("name")
        # Ordinary json.dumps emits NaN when the portable non-finite codec is bypassed.
        return frame.with_columns(pl.lit(json.dumps(float("nan"))).alias("value_json"))

    monkeypatch.setattr(observed_result, "to_recorded_dataframe", faulty_export)
    observed_result.to_parquet(tmp_path)
    with pytest.raises(ArtifactReadError, match="Recorded variable|Recorded value"):
        BacktestResult.from_parquet(tmp_path)
    monkeypatch.setattr(observed_result, "to_recorded_dataframe", original)
    observed_result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_recorded_dataframe()["bar_index"].to_list() == [0, 1, 2]
    assert restored.to_returns_series().to_list() == pytest.approx([-0.001, 0, 0])


@pytest.mark.parametrize("component", ["holdings", "recorded_variables"])
def test_selective_export_marks_omitted_capture_unavailable_in_recovery(
    observed_result, tmp_path, component
):
    selected = [
        "trades",
        "fills",
        "rejected_orders",
        "equity",
        "portfolio_state",
        "metrics",
        "config",
        "spec",
    ]
    observed_result.to_parquet(tmp_path, include=selected)
    recovered = BacktestResult.from_parquet(tmp_path, recovery=True)
    assert any(d.code == "manifest_incomplete" for d in recovered.artifact_diagnostics)
    assert any(
        d.component == component and d.code == "component_unavailable"
        for d in recovered.artifact_diagnostics
    )
    with pytest.raises(ArtifactError, match="not captured|not recorded"):
        if component == "holdings":
            recovered.to_holdings_dataframe()
        else:
            recovered.to_recorded_dataframe()
    assert recovered.to_returns_series().to_list() == pytest.approx([-0.001, 0, 0])


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("account_method", "unrecognized_cash_ledger", "account_method"),
        ("returns_include_commissions", "true", "boolean"),
        ("initial_capital", True, "finite positive"),
    ],
)
def test_reader_rejects_manifest_type_drift_before_interpreting_account_returns(
    observed_result, tmp_path, field, value, reason
):
    observed_result.to_parquet(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactManifestError, match=reason):
        BacktestResult.from_parquet(tmp_path)


def test_empty_engine_export_keeps_empty_observations_and_unavailable_capture(tmp_path):
    class Cash(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    frame = pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "asset": pl.String, "close": pl.Float64}
    )
    result = Engine(DataFeed(prices_df=frame), Cash(), BacktestConfig(calendar=None)).run()
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_equity_dataframe().is_empty()
    assert restored.to_portfolio_state_dataframe().is_empty()
    assert restored.to_daily_returns(calendar=None).is_empty()
    with pytest.raises(ArtifactError, match="not captured"):
        restored.to_holdings_dataframe()
