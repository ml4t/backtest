"""Native Strategy ports must match full published callback/account/fill traces."""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml4t.backtest import DataFeed, Engine, RuntimeServices

_EXAMPLES = Path(__file__).parents[1] / "examples/zipline_native_strategies.py"
_FIXTURES = Path(__file__).parent / "fixtures/zipline_original"
_SPEC = importlib.util.spec_from_file_location("ml4t_native_zipline_examples", _EXAMPLES)
_NATIVE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NATIVE)


def _feed(rows):
    return DataFeed(
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


def _fills(actual, expected):
    assert len(actual) == len(expected)
    for fill, reference in zip(actual, expected, strict=True):
        assert fill.timestamp.isoformat() == reference["timestamp"]
        assert fill.quantity * (1 if fill.side.value == "buy" else -1) == reference["quantity"]
        assert fill.price == pytest.approx(reference["price"], rel=1e-12, abs=1e-10)
        if "asset" in reference:
            assert fill.asset == str(reference["asset"])


def test_native_olmar_matches_all_published_weights_accounts_quantities_basis_and_fills():
    reference = json.loads((_FIXTURES / "reference.json").read_text())
    rows = [
        {
            "timestamp": datetime.fromisoformat(observed["timestamp"]),
            "asset": str(asset),
            **{field: series[index] for field, series in values.items()},
        }
        for asset, values in reference["input"].items()
        for index, observed in enumerate(reference["trace"])
    ]
    assets = [str(i) for i in range(len(reference["symbols"]))]
    runtime = RuntimeServices()
    trace = []

    class Observe(_NATIVE.OnlineMovingAverageReversion):
        def on_data(self, timestamp, data, context, broker):
            super().on_data(timestamp, data, context, broker)
            account = broker.get_account_snapshot()
            positions = [broker.get_position(asset) for asset in assets]
            trace.append(
                {
                    "cash": account.cash,
                    "equity": account.equity,
                    "positions_value": account.positions_value,
                    "quantity": [position.quantity if position else 0 for position in positions],
                    "basis": [
                        position.commission_adjusted_cost_basis if position else 0
                        for position in positions
                    ],
                    "weights": self.weights.tolist(),
                }
            )

    config, limits = _NATIVE.reference_execution(commission_per_share=0)
    config.initial_cash = 10000
    result = Engine(
        _feed(rows),
        Observe(runtime, assets),
        config,
        runtime=runtime,
        execution_limits=limits,
        holdings_sampling="bar",
    ).run()
    for actual, expected in zip(trace, reference["trace"], strict=True):
        for field in ["cash", "equity", "positions_value", "quantity", "basis", "weights"]:
            np.testing.assert_allclose(
                actual[field],
                expected[field],
                rtol=1e-12,
                atol=1e-9,
                err_msg=f"native OLMAR {field} {expected['timestamp']}",
            )
    _fills(result.fills, reference["fills"])


def test_native_dual_ma_preserves_300_session_warmup_both_crossovers_and_recorded_series():
    reference = json.loads((_FIXTURES / "dual-ma-reference.json").read_text())
    rows = [
        {
            "timestamp": datetime.fromisoformat(observed["timestamp"]),
            "asset": "A",
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 100000,
        }
        for price, observed in zip(reference["input_close"], reference["trace"], strict=True)
    ]
    trace = []
    runtime = RuntimeServices()

    class Observe(_NATIVE.DualMovingAverage):
        def on_data(self, timestamp, data, context, broker):
            super().on_data(timestamp, data, context, broker)
            account = broker.get_account_snapshot()
            position = broker.get_position("A")
            trace.append((account.cash, account.equity, position.quantity if position else 0))

    config, limits = _NATIVE.reference_execution()
    result = Engine(
        _feed(rows),
        Observe(runtime, "A"),
        config,
        runtime=runtime,
        execution_limits=limits,
        holdings_sampling="bar",
    ).run()
    np.testing.assert_allclose(
        trace,
        [[row["cash"], row["equity"], row["quantity"]] for row in reference["trace"]],
        rtol=1e-12,
        atol=1e-9,
    )
    from ml4t.backtest.compat.zipline import to_performance_frame

    frame = to_performance_frame(result)
    for name, values in reference["records"].items():
        np.testing.assert_allclose(
            frame[name].to_numpy(dtype=float),
            np.array([np.nan if value is None else value for value in values]),
            rtol=1e-12,
            atol=1e-9,
            equal_nan=True,
        )
    _fills(result.fills, reference["fills"])
    assert frame.iloc[:299].transactions.map(len).eq(0).all()
