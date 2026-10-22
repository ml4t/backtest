"""Actual optional library workflows retain intermediates, account money and exposure units."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from runpy import run_path

import polars as pl
import pytest


@pytest.mark.parametrize("lazy", [False, True])
def test_actual_offline_snapshot_feature_engine_artifact_and_diagnostic_workflow(lazy, tmp_path):
    for module in [
        "ml4t.data.storage",
        "ml4t.engineer.features.preparation",
        "ml4t.diagnostic.integration",
    ]:
        pytest.importorskip(module)
    script = Path(__file__).parents[1] / "examples/offline_migration.py"
    report = run_path(str(script))["run"](tmp_path / "workflow", lazy=lazy)
    assert [row["equity"] for row in report["accounts"]] == [10000, 10000, 10000, 9998, 10008]
    assert [row["cash"] for row in report["accounts"]] == [10000, 10000, 10000, 8968, 8968]
    assert [row["means"]["example:A"] for row in report["callbacks"]] == [
        None,
        None,
        101,
        102,
        None,
    ]
    assert [row["means"]["example:B"] for row in report["callbacks"]] == [None] * 5
    assert [(row["asset"], row["mean"], row["eligible"]) for row in report["features"]] == [
        ("example:A", 101, True),
        ("example:B", 199, False),
        ("example:A", 102, True),
        ("example:B", 198, False),
    ]
    assert [
        (row["asset"], row["timestamp"], row["price"], row["quantity"], row["commission"])
        for row in report["fills"]
    ] == [("example:A", datetime(2025, 2, 6, 21, tzinfo=UTC), 103, 10, 2)]
    assert report["input_provenance"]["consumed_tables"] == {
        "prices": "prices",
        "signals": "features",
    }
    assert report["input_provenance"]["snapshot_id"] == report["prepared_snapshot_id"]
    assert report["holdings_sampling"] == "bar"
    assert [row["equity"] for row in report["consumer_holdings"]] == [
        10000,
        10000,
        10000,
        9998,
        10008,
    ]
    assert [row["gross_exposure"] for row in report["consumer_holdings"]] == [0, 0, 0, 1030, 1040]
    assert [row["date"] for row in report["consumer_holdings"]] == [
        datetime(2025, 2, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(5)
    ]
    assert report["consumer_activity_metrics"]["num_fills"] == 1
    assert report["consumer_activity_metrics"]["total_filled_notional"] == 1030
    assert report["consumer_activity_metrics"]["total_implementation_cost"] == 2
    assert report["consumer_activity_metrics"]["mean_turnover_active"] == pytest.approx(0.103)
    assert report["total_return"] == pytest.approx(0.0008, abs=1e-14)


def test_diagnostic_futures_exposure_does_not_become_principal_cash_or_nav_value():
    integration = pytest.importorskip("ml4t.diagnostic.integration")
    evaluation = pytest.importorskip("ml4t.diagnostic.evaluation")
    if not callable(getattr(evaluation.PortfolioAnalysis, "compute_holdings_analysis", None)):
        pytest.skip(
            "This optional Diagnostic version predates holdings analysis; "
            "the pinned successor integration cohort must execute this workflow"
        )
    from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
    from ml4t.backtest.config import CommissionType

    times = [datetime(2025, 2, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(3)]

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("example:ESH25", 1)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["example:ESH25"] * 3,
                    "close": [4000.0, 4000.0, 4010.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            periods_per_year=252,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        holdings_sampling="bar",
        contract_specs={
            "example:ESH25": ContractSpec(
                "example:ESH25",
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=15000,
                currency="USD",
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
        },
    ).run()
    analysis = integration.portfolio_analysis_from_result(
        result, calendar=None, periods_per_year=252
    )
    timeline = analysis.compute_holdings_analysis().timeline
    assert timeline["cash"].to_list() == [100000, 99998, 100498]
    assert timeline["equity"].to_list() == [100000, 99998, 100498]
    assert timeline["gross_exposure"].to_list() == [0, 200000, 200500]
    assert analysis.compute_summary_stats().total_return == pytest.approx(0.00498, abs=1e-14)
    executions = analysis.compute_transaction_analysis().executions
    assert executions["quantity"].to_list() == [1]
    assert executions["multiplier"].to_list() == [50]
    assert executions["filled_notional"].to_list() == [200000]


def test_actual_zipline_32_portable_result_uses_diagnostic_legacy_adapter_without_runtime(
    monkeypatch,
):
    import builtins
    import json

    import pandas as pd

    legacy = pytest.importorskip("ml4t.diagnostic.integration.zipline")
    from ml4t.specs import FeedSpec

    original_import = builtins.__import__

    def no_legacy_runtime(name, *args, **kwargs):
        if (
            name == "zipline"
            or name.startswith("zipline.")
            or name == "pyfolio"
            or name.startswith("pyfolio.")
        ):
            raise ImportError("Legacy runtimes are deliberately unavailable for the consumer")
        return original_import(name, *args, **kwargs)

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/dated_futures/zipline_3_2.json").read_text()
    )
    case = next(c for c in fixture["cases"] if c["name"] == "expiry")
    source = case["analysis_input"]
    performance = pd.DataFrame.from_records(
        source["records"], index=pd.to_datetime(source["index"])
    )
    metadata = {int(sid): value for sid, value in source["asset_metadata"].items()}
    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", no_legacy_runtime)
        analysis = legacy.portfolio_analysis_from_zipline(
            performance,
            calendar=None,
            periods_per_year=252,
            feed_spec=FeedSpec(
                time_contract_version=2, calendar="us_futures", timestamp_semantics="bar_close"
            ),
            asset_metadata=metadata,
        )
        holdings = analysis.compute_holdings_analysis().timeline
        transactions = analysis.compute_transaction_analysis()
        summary = analysis.compute_summary_stats()
    assert holdings["date"].to_list() == [
        datetime(2025, 7, day, 21, tzinfo=UTC) for day in [7, 8, 9, 10, 11, 14]
    ]
    assert holdings["cash"].to_list() == [100000, 100000, 100500, 101000, 101000, 101000]
    assert holdings["equity"].to_list() == [100000, 100000, 100500, 101000, 101000, 101000]
    assert holdings["gross_exposure"].to_list() == [0, 200500, 201000, 0, 0, 0]
    assert summary.total_return == pytest.approx(0.01, abs=1e-14)
    assert transactions.executions["asset"].to_list() == [17, 17]
    assert transactions.executions["quantity"].to_list() == [1, -1]
    assert transactions.executions["multiplier"].to_list() == [50, 50]
    assert transactions.executions["filled_notional"].to_list() == [200500, 201500]
    assert transactions.metrics["total_implementation_cost"] is None
