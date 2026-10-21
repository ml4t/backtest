"""Migration callbacks must execute on the native ledger with scoped API state."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import DataFeed


def _feed():
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]],
                "asset": ["A"] * 3,
                "close": [100.0, 110.0, 120.0],
                "volume": [1000] * 3,
            }
        ),
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
            "calendar": None,
        },
    )


def test_original_callback_orders_and_record_use_native_fills_fees_and_marked_positions():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.api import (
        get_datetime,
        order,
        record,
        set_commission,
        set_slippage,
        symbol,
    )
    from ml4t.backtest.compat.zipline.assets import Asset
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    events = []

    def initialize(context):
        context.a = symbol("ALPHA")
        context.count = 0
        set_commission(commission.PerShare(cost=0.1, min_trade_cost=1.0))
        set_slippage(slippage.VolumeShareSlippage(volume_limit=1.0, price_impact=0.0))
        events.append("initialize")

    def handle_data(context, data):
        events.append(get_datetime())
        assert data.current(context.a, "price") == [100, 110, 120][context.count]
        if context.count == 0:
            assert context.portfolio.positions[context.a].amount == 0
            order(context.a, 10)
        else:
            assert context.portfolio.positions[context.a].amount == 10
            assert context.portfolio.positions[context.a].cost_basis == 110.1
            assert context.portfolio.cash == 8899
        record(observed_price=data.current(context.a, "price"))
        context.count += 1

    def analyze(context, performance):
        events.append("analyze")
        assert context.count == 3
        assert performance.portfolio_value.tolist() == [10000.0, 9999.0, 10099.0]

    performance = run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "ALPHA", 17)],
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=10000,
    )
    result = performance.attrs["backtest_result"]
    assert result.to_fills_dataframe()["price"].to_list() == [110.0]
    assert result.to_fills_dataframe()["commission"].to_list() == [1.0]
    assert performance.observed_price.tolist() == [100.0, 110.0, 120.0]
    assert performance.returns.tolist() == pytest.approx([0, -0.0001, 100 / 9999])
    assert [len(row) for row in performance.transactions] == [0, 1, 0]
    assert performance.transactions.iloc[1][0]["amount"] == 10
    assert performance.transactions.iloc[1][0]["commission"] == 1
    assert events[0] == "initialize" and events[-1] == "analyze"
    assert len(events) == 5
    with pytest.raises(RuntimeError, match="active migration callback"):
        get_datetime()


def test_failed_and_nested_runs_restore_the_enclosing_callback_api_context():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.api import symbol
    from ml4t.backtest.compat.zipline.assets import Asset

    seen = []

    def outer(context, data):
        before = symbol("ALPHA")

        def fail(context):
            assert symbol("BETA").key == "A"
            raise ValueError("user initialization failed")

        with pytest.raises(ValueError, match="user initialization failed"):
            run_algorithm(feed=_feed(), assets=[Asset("A", "BETA")], initialize=fail)
        seen.append(symbol("ALPHA") == before)

    run_algorithm(feed=_feed(), assets=[Asset("A", "ALPHA")], handle_data=outer)
    assert seen == [True] * 3
    with pytest.raises(RuntimeError, match="active migration callback"):
        symbol("ALPHA")


def test_original_olmar_callbacks_match_published_zipline_ledger_weights_and_fills():
    import importlib.util
    import json
    from pathlib import Path

    import numpy as np

    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.assets import Asset

    root = Path(__file__).parent / "fixtures/zipline_original"
    reference = json.loads((root / "reference.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "ml4t_migration_original_olmar", root / "olmar.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for i, values in reference["input"].items():
        for n, observed in enumerate(reference["trace"]):
            rows.append(
                {
                    "timestamp": datetime.fromisoformat(observed["timestamp"]),
                    "asset": str(i),
                    **{name: column[n] for name, column in values.items()},
                }
            )
    feed = DataFeed(
        prices_df=pl.DataFrame(rows),
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
            "calendar": "NYSE",
            "data_frequency": "1d",
        },
    )
    trace = []

    def initialize(context):
        module.initialize(context, eps=1.05, window_length=5)

    def handle_data(context, data):
        module.handle_data(context, data)
        trace.append(
            {
                "cash": context.portfolio.cash,
                "equity": context.portfolio.portfolio_value,
                "positions_value": context.portfolio.positions_value,
                "quantity": [context.portfolio.positions[a].amount for a in context.sids],
                "basis": [context.portfolio.positions[a].cost_basis for a in context.sids],
                "weights": context.b_t.tolist(),
            }
        )

    performance = run_algorithm(
        feed=feed,
        assets=[Asset(str(i), name, i) for i, name in enumerate(reference["symbols"])],
        initialize=initialize,
        handle_data=handle_data,
        capital_base=10000,
    )
    for actual, expected in zip(trace, reference["trace"], strict=True):
        for field in ["cash", "equity", "positions_value", "quantity", "basis", "weights"]:
            np.testing.assert_allclose(
                actual[field],
                expected[field],
                rtol=1e-12,
                atol=1e-9,
                err_msg=f"OLMAR {field} at {expected['timestamp']}",
            )
    fills = performance.attrs["backtest_result"].fills
    assert len(fills) == len(reference["fills"])
    for actual, expected in zip(fills, reference["fills"], strict=True):
        assert actual.timestamp.isoformat() == expected["timestamp"]
        assert actual.asset == str(expected["asset"])
        assert actual.quantity * (1 if actual.side.value == "buy" else -1) == expected["quantity"]
        assert actual.price == pytest.approx(expected["price"], abs=1e-10, rel=1e-12)


def test_record_first_used_after_initial_bar_keeps_all_native_account_observations():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.assets import Asset

    def initialize(context):
        context.count = 0

    def handle_data(context, data):
        if context.count == 1:
            context.record(late_signal=2.0)
        context.count += 1

    performance = run_algorithm(
        feed=_feed(), assets=[Asset("A", "ALPHA")], initialize=initialize, handle_data=handle_data
    )
    assert len(performance) == 3
    assert performance.late_signal.isna().tolist() == [True, False, False]
    assert performance.late_signal.iloc[1:].tolist() == [2.0, 2.0]


def test_performance_frame_preserves_nanosecond_keys_from_real_native_feed():
    import pandas as pd

    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.assets import Asset

    stamps = pd.date_range("2025-01-06T21:00:00Z", periods=2, freq="ns")
    frame = pl.DataFrame(
        {
            "timestamp": pl.Series(stamps.asi8, dtype=pl.Datetime("ns", "UTC")),
            "asset": ["A", "A"],
            "close": [100.0, 101.0],
            "volume": [10, 10],
        }
    )
    feed = DataFeed(
        prices_df=frame,
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
            "calendar": None,
        },
    )
    performance = run_algorithm(feed=feed, assets=[Asset("A", "ALPHA")])
    assert performance.index.is_unique
    assert performance.index.asi8.tolist() == stamps.asi8.tolist()


@pytest.mark.parametrize(
    "method,fees",
    [("per_share", [1.0, 0.0]), ("per_trade", [2.0, 0.0]), ("per_dollar", [5.5, 6.0])],
)
def test_compiled_equity_commission_methods_charge_actual_partial_order_fees_once(method, fees):
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.assets import Asset
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    def initialize(context):
        models = {
            "per_share": commission.PerShare(0.1, 1),
            "per_trade": commission.PerTrade(2),
            "per_dollar": commission.PerDollar(0.01),
        }
        context.set_commission(models[method])
        context.set_slippage(slippage.VolumeShareSlippage(volume_limit=1.0, price_impact=0.0))
        context.submitted = False

    def handle_data(context, data):
        if not context.submitted:
            context.order(context.symbol("ALPHA"), 10)
            context.submitted = True

    frame = _feed().prices.with_columns(pl.lit(5).alias("volume"))
    feed = DataFeed(prices_df=frame, feed_spec=_feed().feed_spec)
    performance = run_algorithm(
        feed=feed,
        assets=[Asset("A", "ALPHA")],
        initialize=initialize,
        handle_data=handle_data,
        capital_base=10000,
    )
    result = performance.attrs["backtest_result"]
    assert [(f.quantity, f.price) for f in result.fills] == [(5, 110), (5, 120)]
    assert [f.commission for f in result.fills] == pytest.approx(fees)
    assert performance.cash.iloc[-1] == pytest.approx(8850 - sum(fees))
    assert result.config.metadata["execution_limits"] == {
        "model": "shared_observed_bar_volume",
        "max_participation": 1.0,
    }


def test_callback_run_rejects_account_mutation_during_analyze_and_late_cost_reconfiguration():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.assets import Asset
    from ml4t.backtest.compat.zipline.finance import commission

    def handle_data(context, data):
        with pytest.raises(RuntimeError, match="during initialize"):
            context.set_commission(commission.PerTrade(99))

    def analyze(context, performance):
        with pytest.raises(RuntimeError, match="running market callback"):
            context.order(context.symbol("ALPHA"), 1)
        with pytest.raises(RuntimeError, match="running market callback"):
            context.record(too_late=1)

    performance = run_algorithm(
        feed=_feed(), assets=[Asset("A", "ALPHA")], handle_data=handle_data, analyze=analyze
    )
    assert performance.attrs["backtest_result"].fills == []
