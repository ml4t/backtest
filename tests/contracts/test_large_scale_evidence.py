"""Contracts for reconstructable, capability-qualified large-scale evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = Path(__file__).parents[2] / "validation" / "large_scale_evidence.py"
    spec = importlib.util.spec_from_file_location("ml4t_large_scale_evidence", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_workload(module):
    return module.ScaleWorkload(
        name="contract_scale",
        seed=7,
        bars=20,
        assets=3,
        top_n=1,
        bottom_n=1,
        end_session="2024-12-31",
    )


def _complete_report(module, monkeypatch: pytest.MonkeyPatch):
    workload = _small_workload(module)
    monkeypatch.setattr(module, "WORKLOAD", workload)
    module._expected_input_digest.cache_clear()
    config = workload.benchmark_config()
    prices, signals, dates = module.suite.generate_benchmark_data(config, seed=workload.seed)
    digest = module.input_digest(prices, signals, dates)
    records = []
    manifest = module.load_framework_manifest()
    surface_names = ("order_intents", "fills", "trades", "terminal_state")
    scalar_names = ("trade_count", "total_pnl", "final_value")
    for framework in module.FRAMEWORKS:
        target = manifest.targets[framework]
        checks = [
            {
                "name": name,
                "passed": True,
                "expected_sha256": "a" * 64,
                "actual_sha256": "a" * 64,
            }
            for name in surface_names
        ]
        checks.extend(
            {
                "name": name,
                "passed": True,
                "canonical_expected": 1.0,
                "canonical_actual": 1.0,
            }
            for name in scalar_names
        )
        records.append(
            {
                "framework": framework,
                "target": {**target.evidence_metadata(), "actual_version": target.version},
                "python": {"version": "3.12.11", "implementation": "cpython"},
                "ml4t": {"commit": "a" * 40, "dirty": False},
                "source_digests": module._source_digests(),
                "input": {
                    "raw_sha256": digest,
                    "effective_sha256": digest,
                    "conversion": "none",
                },
                "capabilities": {
                    "intents": "canonical strategy trace",
                    "fills": "native",
                    "closed_trades": "native",
                    "terminal_state": "reconstructed",
                    "fill_order": "canonical",
                },
                "comparison": {"passed": True, "checks": checks},
            }
        )
    return module.build_report(records, workload)


def test_input_digest_is_independent_of_datetime_resolution() -> None:
    module = _load_module()
    workload = _small_workload(module)
    config = workload.benchmark_config()
    prices, signals, dates = module.suite.generate_benchmark_data(config, seed=workload.seed)
    expected = module.input_digest(prices, signals, dates)

    converted_prices = {asset: frame.copy() for asset, frame in prices.items()}
    for frame in converted_prices.values():
        frame.index = pd.DatetimeIndex(frame.index.to_numpy(dtype="datetime64[us]"))
    converted_signals = signals.copy()
    converted_signals["timestamp"] = converted_signals["timestamp"].to_numpy(dtype="datetime64[us]")
    converted_dates = pd.DatetimeIndex(dates.to_numpy(dtype="datetime64[us]"))

    assert module.input_digest(converted_prices, converted_signals, converted_dates) == expected


def test_complete_report_reconstructs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    report = _complete_report(module, monkeypatch)

    assert module.report_failures(report, reconstruct_input=True) == []


@pytest.mark.parametrize(
    "mutation",
    ["recipe", "input", "order_intents", "fills", "trades", "terminal_state", "final_value"],
)
def test_large_scale_mutations_fail_closed(mutation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    report = copy.deepcopy(_complete_report(module, monkeypatch))
    if mutation == "recipe":
        report["workload"]["recipe"]["seed"] = 999
    elif mutation == "input":
        report["frameworks"][0]["input"]["raw_sha256"] = "0" * 64
    else:
        check = next(
            check
            for check in report["frameworks"][0]["comparison"]["checks"]
            if check["name"] == mutation
        )
        if mutation == "final_value":
            check["canonical_actual"] = 2.0
        else:
            check["actual_sha256"] = "0" * 64

    assert module.report_failures(report, reconstruct_input=True)


def test_accepted_write_is_atomic(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "accepted.json"
    target.write_text("old\n", encoding="utf-8")

    module._write_json_atomic(target, {"value": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    assert list(tmp_path.iterdir()) == [target]
