"""Tests for the LEAN benchmark adapter helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_benchmark_suite():
    suite_path = Path(__file__).resolve().parents[2] / "validation" / "benchmark_suite.py"
    module_name = "ml4t_validation_benchmark_suite_lean"
    spec = importlib.util.spec_from_file_location(module_name, suite_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_load_lean_artifacts_reads_summary_and_surfaces(tmp_path):
    suite = _load_benchmark_suite()

    summary = {
        "totalPerformance": {
            "tradeStatistics": {"totalNumberOfTrades": "0"},
            "portfolioStatistics": {"endEquity": "$0"},
        },
        "statistics": {"Total Orders": "0", "End Equity": "$0"},
        "state": {"OrderCount": "0"},
    }
    (tmp_path / "result-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "ml4t_symbol_map.json").write_text(
        json.dumps({"AAAA": "AAPL"}),
        encoding="utf-8",
    )
    (tmp_path / "ml4t_daily_equity.csv").write_text(
        "timestamp,equity,cash,total_fees,holdings_value\n"
        "2024-01-02,100000,100000,0,0\n"
        "2024-01-03,100250,25000,4.5,75250\n",
        encoding="utf-8",
    )
    (tmp_path / "ml4t_order_events.csv").write_text(
        "timestamp,symbol,status,direction,fill_quantity,fill_price,fee,message,order_id\n"
        "2024-01-02 00:00:00,AAAA,Filled,Buy,10,100,1.0,,1\n"
        "2024-01-03 00:00:00,AAAA,PartiallyFilled,Sell,-4,101,0.5,partial,2\n"
        "2024-01-03 00:00:00,AAAA,Submitted,Sell,0,0,0.0,pending,2\n",
        encoding="utf-8",
    )

    num_trades, final_value, trades_df, equity_df, order_events_df = suite._load_lean_artifacts(
        tmp_path
    )

    assert num_trades == 2
    assert final_value == 100250.0
    assert trades_df is not None
    assert list(trades_df["asset"]) == ["AAPL", "AAPL"]
    assert list(trades_df["side"]) == ["buy", "sell"]
    assert list(trades_df["quantity"]) == [10, 4]
    assert equity_df is not None
    assert list(equity_df["equity"]) == [100000, 100250]
    assert order_events_df is not None
    assert len(order_events_df) == 3
    assert list(order_events_df["asset"]) == ["AAPL", "AAPL", "AAPL"]


def test_load_lean_artifacts_requires_summary(tmp_path):
    suite = _load_benchmark_suite()

    try:
        suite._load_lean_artifacts(tmp_path)
    except FileNotFoundError as exc:
        assert "summary file not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError when LEAN summary is missing")
