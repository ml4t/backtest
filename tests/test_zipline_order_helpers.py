"""Imported order helpers preserve signed rounding, pending targets and native styles."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import DataFeed


def _feed():
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8, 10]],
                "asset": ["A"] * 4,
                "close": [100.0] * 4,
                "volume": [1000] * 4,
            }
        ),
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
            "calendar": None,
        },
    )


@pytest.mark.parametrize("kind", ["target", "target_value", "target_percent"])
def test_target_helpers_subtract_held_quantity_before_rounding_and_ignore_pending_targets(kind):
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.assets import Asset
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    def initialize(context):
        context.n = 0
        context.a = context.symbol("ALPHA")
        context.set_commission(commission.PerShare(0, 0))
        context.set_slippage(slippage.FixedSlippage(0))

    def handle_data(context, data):
        if context.n == 0:
            context.order(context.a, 6)
        if context.n == 1:
            helpers = {
                "target": (context.order_target, 5.5),
                "target_value": (context.order_target_value, 550),
                "target_percent": (context.order_target_percent, 0.055),
            }
            call, value = helpers[kind]
            assert call(context.a, value) is None
            assert call(context.a, 0) is not None
            assert call(context.a, 0) is not None
        context.n += 1

    performance = run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "ALPHA")],
        initialize=initialize,
        handle_data=handle_data,
        capital_base=10000,
    )
    fills = performance.attrs["backtest_result"].fills
    assert [f.quantity * (1 if f.side.value == "buy" else -1) for f in fills] == [6, -6, -6]
    assert performance.attrs["backtest_result"].trades[-1].quantity == -6


def test_order_value_and_style_wrappers_use_native_prices_and_keep_final_limit_bounds():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.api import order_value
    from ml4t.backtest.compat.zipline.assets import Asset
    from ml4t.backtest.compat.zipline.finance import commission, slippage
    from ml4t.backtest.compat.zipline.finance.execution import LimitOrder

    def initialize(context):
        context.n = 0
        context.set_commission(commission.PerShare(0, 0))
        context.set_slippage(slippage.FixedSlippage(spread=2))

    def handle_data(context, data):
        if context.n == 0:
            a = context.symbol("ALPHA")
            order_value(a, 299.995, style=LimitOrder(101))
            order_value(a, 100, style=LimitOrder(100))
        context.n += 1

    performance = run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "ALPHA")],
        initialize=initialize,
        handle_data=handle_data,
        capital_base=10000,
    )
    fills = performance.attrs["backtest_result"].fills
    assert [(f.quantity, f.price, f.limit_price) for f in fills] == [(3, 101, 101)]


def test_original_dual_moving_average_keeps_300_day_warmup_records_and_both_crossovers():
    import importlib.util
    import json
    from pathlib import Path

    import numpy as np

    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.assets import Asset

    root = Path(__file__).parent / "fixtures/zipline_original"
    reference = json.loads((root / "dual-ma-reference.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "ml4t_migration_dual_ma", root / "dual_moving_average.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    times = [datetime.fromisoformat(row["timestamp"]) for row in reference["trace"]]
    close = reference["input_close"]
    feed = DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": times,
                "asset": ["A"] * len(times),
                "close": close,
                "open": close,
                "high": [p + 1 for p in close],
                "low": [p - 1 for p in close],
                "volume": [100000] * len(times),
            }
        ),
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "calendar": "NYSE",
            "data_frequency": "1d",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )
    trace = []

    def handle_data(context, data):
        module.handle_data(context, data)
        trace.append(
            (
                context.portfolio.cash,
                context.portfolio.portfolio_value,
                context.portfolio.positions[context.sym].amount,
            )
        )

    performance = run_algorithm(
        feed=feed,
        assets=[Asset("A", "AAPL", 0)],
        initialize=module.initialize,
        handle_data=handle_data,
        capital_base=100000,
    )
    assert len(trace) == 350
    np.testing.assert_allclose(
        trace,
        [[r["cash"], r["equity"], r["quantity"]] for r in reference["trace"]],
        rtol=1e-12,
        atol=1e-9,
    )
    for name, values in reference["records"].items():
        np.testing.assert_allclose(
            performance[name].to_numpy(dtype=float),
            np.array([np.nan if v is None else v for v in values]),
            rtol=1e-12,
            atol=1e-9,
            equal_nan=True,
        )
    fills = performance.attrs["backtest_result"].fills
    assert len(fills) == len(reference["fills"])
    for actual, expected in zip(fills, reference["fills"], strict=True):
        assert actual.timestamp.isoformat() == expected["timestamp"]
        assert actual.quantity * (1 if actual.side.value == "buy" else -1) == expected["quantity"]
        assert actual.price == pytest.approx(expected["price"], abs=1e-10, rel=1e-12)
    assert performance.iloc[:299].transactions.map(len).eq(0).all()
