"""Run offline Data preparation, Engineer features, Backtest and Diagnostic.

This synthetic equity example has no network calls during preparation or execution.
Install the qualified source builds described in the migration guide before running it.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from ml4t.backtest import (
    BacktestConfig,
    BacktestResult,
    ContractSpec,
    DataFeed,
    Engine,
    Strategy,
)
from ml4t.backtest.config import CommissionType, ExecutionMode, ExecutionPrice


def run(output: Path, *, lazy: bool = False) -> dict[str, Any]:
    from ml4t.data.adjustments import prepare_adjusted_history
    from ml4t.data.storage import SnapshotTable, export_snapshot, read_snapshot
    from ml4t.diagnostic.integration import portfolio_analysis_from_result
    from ml4t.engineer.artifacts import FeatureDefinition, FeatureSchema, FeatureSpec
    from ml4t.engineer.features.preparation import prepare_feature_panel
    from ml4t.engineer.features.rolling import finite_mean
    from ml4t.specs import FeedSpec, WindowUnit

    times = [datetime(2025, 2, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(5)]
    feed_spec = FeedSpec(
        time_contract_version=2,
        calendar=None,
        timestamp_semantics="event_time",
        entity_col="asset",
        data_frequency="daily",
    )
    prices = pl.DataFrame(
        [
            {
                "timestamp": timestamp,
                "asset": asset,
                "close": float(price),
                "available_at": timestamp,
                "source": "declared-example",
                "revision_id": "v1",
            }
            for i, timestamp in enumerate(times)
            for asset, price in [("example:A", 100 + i), ("example:B", 200 - i)]
        ]
    )
    universe = pl.DataFrame(
        {
            "asset": ["example:A", "example:B"],
            "effective_from": [times[0] - timedelta(days=1)] * 2,
            "effective_to": pl.Series([None, None], dtype=pl.Datetime("us", "UTC")),
            "available_at": [times[0] - timedelta(days=1)] * 2,
            "member": [True, False],
        }
    )
    decisions = pl.DataFrame(
        {"timestamp": times[2:4], "decision_time": times[2:4], "universe_at": times[2:4]}
    )
    config = BacktestConfig(
        calendar=None,
        initial_cash=10000,
        periods_per_year=252,
        execution_mode=ExecutionMode.NEXT_BAR,
        next_bar_execution_price=ExecutionPrice.CLOSE,
        commission_type=CommissionType.PER_TRADE,
        commission_per_trade=2,
        slippage_rate=0,
    )
    declarations = {
        "source_revision": "offline-migration-example-v1",
        "identity_namespace": "example",
        "units": {"close": "USD/share"},
        "as_of": times[-1].isoformat(),
        "price_basis": "raw",
        "share_basis": "source",
        "revisions": {"example": "v1"},
    }
    effective = {
        "strategy": "eligible-three-observation-momentum-v1",
        "random_seed": None,
        "backtest": config.to_dict(),
        "holdings_sampling": "bar",
        "feature_input_mode": "lazy" if lazy else "eager",
    }
    raw = export_snapshot(
        output / "raw-inputs",
        {
            "prices": SnapshotTable(
                prices, role="prices", metadata=declarations, feed_spec=feed_spec
            ),
            "universe": SnapshotTable(universe, role="universe", metadata=declarations),
            "decisions": SnapshotTable(decisions, role="decisions", metadata=declarations),
        },
        effective_config=effective,
        artifacts={},
        calendar=None,
    )
    captured = read_snapshot(output / "raw-inputs")
    histories = []
    history_metadata = []
    for decision in times[2:4]:
        prepared = prepare_adjusted_history(
            captured.tables["prices"],
            None,
            as_of=decision,
            feed_spec=feed_spec,
            basis="raw",
            action_coverage="complete",
        )
        histories.append(prepared.prices.with_columns(pl.lit(decision).alias("feature_decision")))
        history_metadata.append(prepared.metadata)
    history = pl.concat(histories)
    features = prepare_feature_panel(
        history.lazy() if lazy else history,
        features={"mean": finite_mean("close", period=3)},
        lookbacks={"mean": 3},
        input_columns=["close"],
        decisions=captured.tables["decisions"],
        universe=captured.tables["universe"],
        price_basis="raw",
        feed_spec=feed_spec,
        window_unit="observation",
        universe_time_col="universe_at",
        history_decision_time_col="feature_decision",
    )
    features = features.collect() if isinstance(features, pl.LazyFrame) else features
    feature_spec = FeatureSpec(
        artifact_id="offline-example-mean3",
        schema=FeatureSchema(
            feature_columns=("mean",),
            available_at_col="available_at",
            decision_time_col="decision_time",
            eligible_col="eligible",
            universe_time_col="universe_at",
        ),
        definition=FeatureDefinition(
            feed_spec=feed_spec,
            window_unit=WindowUnit.OBSERVATION,
            price_basis="raw",
            lookbacks={"mean": 3},
            missing_input_policy="complete",
            source_artifacts=(raw.snapshot_id,),
        ),
    )
    prepared_inputs = export_snapshot(
        output / "prepared-inputs",
        {
            "prices": SnapshotTable(
                captured.tables["prices"], role="prices", metadata=declarations, feed_spec=feed_spec
            ),
            "features": SnapshotTable(
                features,
                role="features",
                metadata={**declarations, "units": {"mean": "USD/share"}},
                artifact=feature_spec,
            ),
            "universe": SnapshotTable(
                captured.tables["universe"], role="universe", metadata=declarations
            ),
            "decisions": SnapshotTable(
                captured.tables["decisions"], role="decisions", metadata=declarations
            ),
        },
        effective_config=effective,
        artifacts={"raw_inputs": raw.snapshot_id},
        calendar=None,
    )
    feed = DataFeed.from_snapshot(output / "prepared-inputs", signals_table="features")
    seen = []

    class Momentum(Strategy):
        def on_data(self, timestamp, data, context, broker):
            row = {asset: values["signals"].get("mean") for asset, values in data.items()}
            seen.append({"timestamp": timestamp.isoformat(), "means": row})
            for asset, values in data.items():
                mean = values["signals"].get("mean")
                if (
                    mean is not None
                    and values["price"] > mean
                    and broker.get_position(asset) is None
                ):
                    broker.submit_order(asset, 10)

    result = Engine(
        feed,
        Momentum(),
        config,
        contract_specs={
            asset: ContractSpec(asset, currency="USD") for asset in ["example:A", "example:B"]
        },
        holdings_sampling="bar",
    ).run()
    result.to_parquet(output / "result")
    restored = BacktestResult.from_parquet(output / "result")
    analysis = portfolio_analysis_from_result(restored, calendar=None, periods_per_year=252)
    stats = analysis.compute_summary_stats()
    holdings = analysis.compute_holdings_analysis()
    transactions = analysis.compute_transaction_analysis()
    report = {
        "installed_versions": {
            name: version(name)
            for name in [
                "ml4t-data",
                "ml4t-engineer",
                "ml4t-backtest",
                "ml4t-diagnostic",
                "ml4t-specs",
            ]
        },
        "raw_snapshot_id": raw.snapshot_id,
        "prepared_snapshot_id": prepared_inputs.snapshot_id,
        "input_provenance": restored.input_provenance,
        "effective_backtest_config": restored.config.to_dict(),
        "holdings_sampling": restored.holdings_sampling,
        "preparation": history_metadata,
        "features": features.to_dicts(),
        "callbacks": seen,
        "accounts": restored.to_portfolio_state_dataframe().to_dicts(),
        "fills": restored.to_fills_dataframe().to_dicts(),
        "consumer_holdings": holdings.timeline.to_dicts(),
        "consumer_executions": transactions.executions.to_dicts(),
        "consumer_activity": transactions.timeline.to_dicts(),
        "consumer_cost_scenarios": transactions.scenarios.to_dicts(),
        "consumer_activity_metrics": transactions.metrics,
        "consumer_activity_metadata": transactions.metadata,
        "total_return": stats.total_return,
    }
    (output / "workflow.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lazy", action="store_true")
    args = parser.parse_args()
    report = run(args.output, lazy=args.lazy)
    print(
        f"observations={len(report['accounts'])} fills={len(report['fills'])} total_return={report['total_return']:.6f}"
    )
