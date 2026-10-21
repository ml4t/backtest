"""Saved native accounts must preserve migration records and actual execution identities."""

from datetime import UTC, datetime

import pandas as pd
import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset
from ml4t.backtest.compat.zipline.finance import commission, slippage
from ml4t.backtest.config import CommissionType, ExecutionMode
from ml4t.backtest.result import BacktestResult


def _feed():
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]],
                "asset": ["A"] * 3,
                "close": [100.0, 110.0, 120.0],
                "volume": [1000.0] * 3,
            }
        ),
        feed_spec={
            "calendar": None,
            "timestamp_semantics": "event_time",
            "time_contract_version": 2,
        },
    )


def _migrated():
    ids = []

    def initialize(context):
        context.a = context.symbol("ALPHA")
        context.set_commission(commission.PerShare(0, 0))
        context.set_slippage(slippage.FixedSlippage(0))
        context.n = 0

    def handle(context, data):
        if context.n == 0:
            ids.append(context.order(context.a, 2))
        else:
            context.record(signal={"value": context.n, "eligible": True})
        context.n += 1

    return run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "ALPHA", 17)],
        initialize=initialize,
        handle_data=handle,
        capital_base=1000,
    ), ids


def test_saved_migration_frame_preserves_scoped_ids_assets_late_records_and_first_observation(
    tmp_path,
):
    original, ids = _migrated()
    native = original.attrs["backtest_result"]
    native.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    from ml4t.backtest.compat.zipline import to_performance_frame

    projected = to_performance_frame(restored)
    pd.testing.assert_frame_equal(original, projected)
    assert projected.attrs["backtest_result"] is restored
    assert projected["cash"].tolist() == [1000.0, 780.0, 780.0]
    assert projected["portfolio_value"].tolist() == [1000.0, 1000.0, 1020.0]
    assert projected["returns"].tolist() == pytest.approx([0.0, 0.0, 0.02])
    assert pd.isna(projected["signal"].iloc[0])
    assert projected["signal"].iloc[1] == {"value": 1, "eligible": True}
    fill = projected["transactions"].iloc[1][0]
    assert fill["sid"].key == "A" and fill["sid"].symbol == "ALPHA"
    assert int(fill["sid"]) == 17
    assert fill["order_id"] == ids[0]
    assert fill["native_order_id"] == "ORD-1"
    assert fill["amount"] == 2 and fill["price"] == 110 and fill["commission"] == 0


def test_native_result_conversion_keeps_unknown_symbols_unknown_and_reports_real_first_fee(
    tmp_path,
):
    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 6:
                broker.submit_order("A", 2)
            elif timestamp.day == 7:
                broker.submit_order("A", -1)

    result = Engine(
        _feed(),
        Trade(),
        BacktestConfig(
            calendar=None,
            initial_cash=1000,
            execution_mode=ExecutionMode.SAME_BAR,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=1,
        ),
    ).run()
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    from ml4t.backtest.compat.zipline import to_performance_frame

    frame = to_performance_frame(restored)
    assert frame["cash"].tolist() == [799.0, 908.0, 908.0]
    assert frame["portfolio_value"].tolist() == [999.0, 1018.0, 1028.0]
    assert frame["returns"].tolist() == pytest.approx([-0.001, 1018 / 999 - 1, 1028 / 1018 - 1])
    trades = [item for group in frame.transactions for item in group]
    assert [item["amount"] for item in trades] == [2, -1]
    assert [item["commission"] for item in trades] == [1, 1]
    assert trades[0]["sid"].key == "A" and trades[0]["sid"].symbol is None
    assert [item["order_id"] for item in trades] == ["ORD-1", "ORD-2"]
    explicit = to_performance_frame(restored, assets=[Asset("A", "ALPHA", 17)])
    assert int(explicit.transactions.iloc[0][0]["sid"]) == 17


def test_projection_refuses_incomplete_or_ambiguous_explicit_identity_mapping():
    original, _ = _migrated()
    from ml4t.backtest.compat.zipline import to_performance_frame

    result = original.attrs["backtest_result"]
    with pytest.raises(ValueError, match="mapping.*A"):
        to_performance_frame(result, assets=[])
    with pytest.raises(ValueError, match="Duplicate"):
        to_performance_frame(result, assets=[Asset("A"), Asset("A", "OTHER")])
    with pytest.raises(ValueError, match="Ambiguous legacy SID"):
        to_performance_frame(result, assets=[Asset("A", legacy_sid=17), Asset("B", legacy_sid=17)])


def test_projection_preserves_nanosecond_fill_grouping_after_real_artifact_reload(tmp_path):
    from ml4t.backtest.compat.zipline import to_performance_frame

    epoch = pd.Timestamp("2025-01-06T21:00:00Z").value
    stamps = pl.Series("timestamp", [epoch + i for i in [1, 2, 3]], dtype=pl.Int64).cast(
        pl.Datetime("ns", "UTC")
    )
    feed = DataFeed(
        prices_df=pl.DataFrame({"timestamp": stamps, "asset": ["A"] * 3, "close": [100.0] * 3}),
        feed_spec={
            "calendar": None,
            "timestamp_semantics": "event_time",
            "time_contract_version": 2,
        },
    )

    class Trade(Strategy):
        def on_start(self, broker):
            self.n = 0

        def on_data(self, timestamp, data, context, broker):
            if self.n < 2:
                broker.submit_order("A", 1 if self.n == 0 else -1)
            broker.record(counter=self.n)
            self.n += 1

    result = Engine(
        feed,
        Trade(),
        BacktestConfig(calendar=None, initial_cash=1000, execution_mode=ExecutionMode.SAME_BAR),
    ).run()
    result.to_parquet(tmp_path)
    frame = to_performance_frame(BacktestResult.from_parquet(tmp_path))
    assert frame.index.as_unit("ns").asi8.tolist() == [epoch + i for i in [1, 2, 3]]
    assert [len(group) for group in frame.transactions] == [1, 1, 0]
    assert [frame.transactions.iloc[i][0]["dt"].value for i in [0, 1]] == [epoch + 1, epoch + 2]
    assert frame.counter.tolist() == [0, 1, 2]


def test_mutating_projected_record_does_not_mutate_saved_native_observation():
    from ml4t.backtest.compat.zipline import to_performance_frame

    frame, _ = _migrated()
    native = frame.attrs["backtest_result"]
    frame.signal.iloc[1]["value"] = 999
    repeated = to_performance_frame(native)
    assert repeated.signal.iloc[1] == {"value": 1, "eligible": True}
