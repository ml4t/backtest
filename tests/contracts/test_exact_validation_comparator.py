"""Exact release-comparator contracts for scenario and benchmark parity."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parents[2]
_VALIDATION_DIR = _PROJECT_ROOT / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

from common.comparator import compare_results  # noqa: E402
from common.ml4t_runner import run_ml4t  # noqa: E402
from common.provenance import generate_inputs  # noqa: E402
from common.types import FrameworkResult  # noqa: E402
from scenarios.definitions import SCENARIOS  # noqa: E402


def _load_benchmark_suite():
    module_name = "ml4t_exact_comparator_benchmark_suite"
    spec = importlib.util.spec_from_file_location(
        module_name, _VALIDATION_DIR / "benchmark_suite.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _trade(*, offset: int = 0) -> dict[str, object]:
    entry_time = pd.Timestamp("2024-01-02") + pd.Timedelta(days=offset)
    return {
        "entry_time": entry_time,
        "exit_time": entry_time + pd.Timedelta(days=1),
        "asset": "AAPL",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "pnl": 10.0,
        "size": 10.0,
        "direction": "Long",
        "commission": 0.0,
    }


def _fill(*, offset: int = 0) -> dict[str, object]:
    return {
        "timestamp": pd.Timestamp("2024-01-02") + pd.Timedelta(days=offset),
        "asset": "AAPL",
        "side": "buy" if offset == 0 else "sell",
        "quantity": 10.0,
        "price": 100.0 + offset,
        "commission": 0.0,
    }


def test_zipline_nine_dollar_difference_fails_despite_diagnostic_tolerance() -> None:
    expected = FrameworkResult("Zipline", 100_000.0, 0.0, 1, trades=[_trade()])
    actual = FrameworkResult(
        "ml4t.backtest",
        99_990.54,
        -9.46,
        1,
        trades=[_trade()],
    )

    comparison = compare_results(SCENARIOS["15"], expected, actual)
    checks = {check.name: check for check in comparison.checks}

    assert comparison.passed is False
    assert checks["final_value"].passed is False
    assert checks["total_pnl"].passed is False
    assert "diagnostic_limit=$10.0" in checks["total_pnl"].message


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("entry_time", pd.Timestamp("2024-01-03")),
        ("exit_time", pd.Timestamp("2024-01-04")),
        ("asset", "MSFT"),
        ("direction", "short"),
        ("size", 10.00000001),
        ("entry_price", 100.00000001),
        ("exit_price", 101.00000001),
        ("pnl", 10.00000001),
        ("commission", 0.00000001),
    ],
)
def test_scenario_trade_fields_require_exact_equality(field: str, changed: object) -> None:
    expected = FrameworkResult("Backtrader", 100_010.0, 10.0, 1, trades=[_trade()])
    changed_trade = _trade()
    changed_trade[field] = changed
    actual = FrameworkResult(
        "ml4t.backtest",
        100_010.0,
        10.0,
        1,
        trades=[changed_trade],
    )

    comparison = compare_results(SCENARIOS["01"], expected, actual)

    assert comparison.passed is False
    trade_check = next(check for check in comparison.checks if check.name == "trade_level_match")
    assert trade_check.passed is False
    assert field in trade_check.message


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("timestamp", pd.Timestamp("2024-01-03")),
        ("asset", "MSFT"),
        ("side", "sell"),
        ("quantity", 10.00000001),
        ("price", 100.00000001),
        ("commission", 0.00000001),
    ],
)
def test_scenario_fill_fields_require_exact_equality(field: str, changed: object) -> None:
    expected = FrameworkResult("Backtrader", 100_010.0, 10.0, 1, trades=[_trade()], fills=[_fill()])
    changed_fill = _fill()
    changed_fill[field] = changed
    actual = FrameworkResult(
        "ml4t.backtest",
        100_010.0,
        10.0,
        1,
        trades=[_trade()],
        fills=[changed_fill],
    )

    comparison = compare_results(SCENARIOS["01"], expected, actual)
    fill_check = next(check for check in comparison.checks if check.name == "fill_level_match")

    assert comparison.passed is False
    assert fill_check.passed is False
    assert field in fill_check.message


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "ordering", "malformed"])
def test_scenario_record_shape_negative_controls(mutation: str) -> None:
    expected_trades = [_trade(offset=0), _trade(offset=2)]
    actual_trades = copy.deepcopy(expected_trades)
    if mutation == "missing":
        actual_trades[0].pop("asset")
    elif mutation == "duplicate":
        actual_trades.append(copy.deepcopy(actual_trades[-1]))
    elif mutation == "ordering":
        actual_trades.reverse()
    else:
        actual_trades[0]["pnl"] = float("nan")
    expected = FrameworkResult("Backtrader", 100_020.0, 20.0, 2, trades=expected_trades)
    actual = FrameworkResult("ml4t.backtest", 100_020.0, 20.0, 2, trades=actual_trades)

    comparison = compare_results(SCENARIOS["01"], expected, actual)
    trade_check = next(check for check in comparison.checks if check.name == "trade_level_match")

    assert comparison.passed is False
    assert trade_check.passed is False


def test_wrong_execution_profile_remains_a_detected_negative_control() -> None:
    scenario = SCENARIOS["01"]
    prices, entries, exits = generate_inputs(scenario, "backtrader")
    reference = run_ml4t(scenario, prices, entries, exits, framework="backtrader")
    reference.framework = "Backtrader reference"
    wrong_profile = run_ml4t(scenario, prices, entries, exits, framework="vectorbt_oss")

    comparison = compare_results(scenario, reference, wrong_profile)

    assert comparison.passed is False
    assert any(not check.passed for check in comparison.checks)


def _benchmark_result(suite, framework: str):
    timestamp = pd.Timestamp("2024-01-02")
    return suite.BenchmarkResult(
        framework=framework,
        scenario="Exact comparison",
        runtime_sec=1.0,
        num_trades=1,
        final_value=100_010.0,
        memory_mb=1.0,
        fills_df=pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "asset": "AAPL",
                    "side": "buy",
                    "quantity": 10.0,
                    "price": 100.0,
                    "commission": 0.0,
                },
                {
                    "timestamp": timestamp + pd.Timedelta(days=1),
                    "asset": "AAPL",
                    "side": "sell",
                    "quantity": 10.0,
                    "price": 101.0,
                    "commission": 0.0,
                },
            ]
        ),
        trades_df=pd.DataFrame(
            [
                {
                    "entry_time": timestamp,
                    "exit_time": timestamp + pd.Timedelta(days=1),
                    "asset": "AAPL",
                    "side": "long",
                    "quantity": 10.0,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "pnl": 10.0,
                }
            ]
        ),
        target_trace_df=pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "asset": "AAPL",
                    "prev_target": 0.0,
                    "target": 10.0,
                    "delta": 10.0,
                    "action": "open",
                }
            ]
        ),
    )


def test_zero_gap_benchmark_passes_and_writes_deterministic_artifact(tmp_path: Path) -> None:
    suite = _load_benchmark_suite()
    expected = _benchmark_result(suite, "Reference")
    actual = _benchmark_result(suite, "ml4t.backtest")

    first = suite.compare_benchmark_results_exact(expected, actual, initial_cash=100_000.0)
    second = suite.compare_benchmark_results_exact(expected, actual, initial_cash=100_000.0)
    output = tmp_path / "exact-comparison.json"
    suite.write_exact_comparison_artifact(first, output)

    assert first == second
    assert first["passed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert all(check["passed"] for check in first["checks"])


def test_fixed_point_zero_fill_and_trade_records_are_omitted() -> None:
    suite = _load_benchmark_suite()
    timestamp = pd.Timestamp("2024-01-02")
    dust_fill = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "asset": "AAPL",
                "side": "buy",
                "quantity": 1e-12,
                "price": 100.0,
                "commission": 0.0,
            }
        ]
    )
    dust_trade = pd.DataFrame(
        [
            {
                "entry_time": timestamp,
                "exit_time": timestamp,
                "asset": "AAPL",
                "side": "long",
                "quantity": 1e-12,
                "entry_price": 100.0,
                "exit_price": 100.0,
                "pnl": 0.0,
            }
        ]
    )

    assert suite.canonical_fill_records(dust_fill) == []
    assert suite.canonical_trade_records(dust_trade) == []


def test_daily_records_normalize_framework_session_timestamp_and_asset_wrapper() -> None:
    suite = _load_benchmark_suite()
    frame = pd.DataFrame(
        [
            {
                "dt": pd.Timestamp("2024-01-02 21:00:00", tz="UTC"),
                "asset": "Equity(1 [AAPL])",
                "amount": 10.0,
                "price": 100.0,
                "commission": 0.0,
            }
        ]
    )

    records = suite.canonical_fill_records(frame, timestamp_domain="session_date")

    assert records == [
        {
            "timestamp": "2024-01-02",
            "asset": "AAPL",
            "side": "buy",
            "quantity": 10.0,
            "price": 100.0,
            "commission": 0.0,
        }
    ]


def test_target_trace_excludes_unchanged_targets() -> None:
    suite = _load_benchmark_suite()
    index = pd.date_range("2024-01-02", periods=3, freq="D")
    targets = pd.DataFrame({"AAPL": [10.0, 10.0, 0.0], "MSFT": [0.0, 5.0, 5.0]}, index=index)

    trace = suite.build_canonical_target_trace(targets)

    assert trace[["timestamp", "asset", "prev_target", "target"]].to_dict("records") == [
        {
            "timestamp": index[0],
            "asset": "AAPL",
            "prev_target": 0.0,
            "target": 10.0,
        },
        {
            "timestamp": index[1],
            "asset": "MSFT",
            "prev_target": 0.0,
            "target": 5.0,
        },
        {
            "timestamp": index[2],
            "asset": "AAPL",
            "prev_target": 10.0,
            "target": 0.0,
        },
    ]


def test_terminal_value_uses_exact_microdollar_representation() -> None:
    suite = _load_benchmark_suite()
    expected = _benchmark_result(suite, "Reference")
    same_microdollar = copy.deepcopy(_benchmark_result(suite, "ml4t.backtest"))
    same_microdollar.final_value += 4e-8
    different_microdollar = copy.deepcopy(_benchmark_result(suite, "ml4t.backtest"))
    different_microdollar.final_value += 1e-6

    same = suite.compare_benchmark_results_exact(expected, same_microdollar, initial_cash=100_000.0)
    different = suite.compare_benchmark_results_exact(
        expected, different_microdollar, initial_cash=100_000.0
    )

    assert same["passed"] is True
    assert different["passed"] is False


def _replace_column(result, attribute: str, column: str, values: list[float]) -> None:
    frame = getattr(result, attribute)
    frame.loc[:, column] = values


@pytest.mark.parametrize(
    ("check_name", "mutate"),
    [
        (
            "order_intents",
            lambda result: _replace_column(result, "target_trace_df", "target", [11.0]),
        ),
        (
            "fills",
            lambda result: _replace_column(result, "fills_df", "price", [100.0, 101.01]),
        ),
        (
            "trades",
            lambda result: _replace_column(result, "trades_df", "pnl", [9.99]),
        ),
        ("trade_count", lambda result: setattr(result, "num_trades", 2)),
        ("total_pnl", lambda result: setattr(result, "final_value", 100_019.46)),
    ],
)
def test_selected_benchmark_comparator_fails_each_covered_divergence(
    check_name: str,
    mutate,
) -> None:
    suite = _load_benchmark_suite()
    expected = _benchmark_result(suite, "Reference")
    actual = copy.deepcopy(_benchmark_result(suite, "ml4t.backtest"))
    mutate(actual)

    artifact = suite.compare_benchmark_results_exact(expected, actual, initial_cash=100_000.0)
    checks = {check["name"]: check for check in artifact["checks"]}

    assert artifact["passed"] is False
    assert checks[check_name]["passed"] is False
