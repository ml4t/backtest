"""Contracts for repeated cross-framework performance evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validation import cross_framework_performance as performance


def _sample(runner: performance.RunnerSpec, index: int = 0) -> dict:
    target = performance.load_framework_manifest().targets[runner.framework]
    target_metadata = {**target.evidence_metadata(), "actual_version": target.version}
    if runner.framework == "vectorbt_pro":
        target_metadata.update(
            {
                "actual_commit": target.source_commit,
                "actual_immutable_id": target.immutable_id,
            }
        )
    input_sha256 = performance.scale._expected_input_digest(performance.WORKLOAD)
    output = {
        "order_intents": {"count": 1, "sha256": "a" * 64},
        "fills": {"count": 1, "sha256": "b" * 64},
        "trades": {"count": 1, "sha256": "c" * 64},
        "trade_count": 1,
        "total_pnl": 1.0,
        "final_value": 1_000_001.0,
        "terminal_state_sha256": "d" * 64,
    }
    return {
        "runner_id": runner.runner_id,
        "framework": runner.framework,
        "side": runner.side,
        "profile": runner.profile,
        "target": target_metadata,
        "python": {"implementation": "cpython", "version": "3.12.11"},
        "ml4t": {"commit": "a" * 40, "dirty": False},
        "input": {"raw_sha256": input_sha256, "effective_sha256": input_sha256},
        "stages_seconds": {
            "input_generation": 0.01,
            "framework_call": 0.1 + index / 1_000,
            "adapter_reported_engine": 0.08 + index / 1_000,
            "output_validation": 0.01,
            "worker_total": 0.12 + index / 1_000,
        },
        "process_peak_rss_mb": 100.0 + index,
        "worker_self_peak_rss_mb": 90.0 + index,
        "process_wall_seconds": 0.2 + index / 100,
        "lean_container_observed": True,
        "thread_environment": performance.THREAD_ENVIRONMENT,
        "output": output,
        "output_sha256": performance._json_digest(output),
        "captured_log_tail": "",
    }


def test_bootstrap_interval_is_reproducible_and_requires_ten_samples() -> None:
    values = [float(value) for value in range(10)]

    first = performance.bootstrap_median_interval(values, seed=7)
    second = performance.bootstrap_median_interval(values, seed=7)

    assert first == second
    assert first[0] <= 4.5 <= first[1]
    with pytest.raises(ValueError, match="At least 10"):
        performance.bootstrap_median_interval(values[:9], seed=7)


def test_instrumentation_calibration_measures_child_wall_time_and_rss() -> None:
    result = performance.run_calibration_process()

    assert result["allocated_bytes"] == 32 * 1024 * 1024
    assert result["worker_seconds"] >= 0.02
    assert result["process_wall_seconds"] >= result["worker_seconds"]
    assert result["process_peak_rss_mb"] >= 20


def test_protocol_identity_changes_with_setup_or_cache_boundary() -> None:
    original = performance._json_digest(performance.MEASUREMENT_PROTOCOL)
    changed = copy.deepcopy(performance.MEASUREMENT_PROTOCOL)
    changed["cache_policy"] = "different"

    assert performance._json_digest(changed) != original


def test_complete_report_recomputes_raw_sample_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = {
        "allocated_bytes": 32 * 1024 * 1024,
        "worker_seconds": 0.025,
        "process_wall_seconds": 0.03,
        "process_peak_rss_mb": 40.0,
    }
    monkeypatch.setattr(performance, "run_calibration_process", lambda: calibration)
    calls = {runner.runner_id: 0 for runner in performance.RUNNERS}

    def run(runner: performance.RunnerSpec) -> dict:
        index = calls[runner.runner_id]
        calls[runner.runner_id] += 1
        return _sample(runner, max(index - 1, 0))

    monkeypatch.setattr(performance, "_run_process", run)

    report = performance.collect_evidence(10)

    assert performance.report_failures(report) == []
    assert report["controlled"]["passed"] is True
    assert report["idiomatic"]["equivalence_claim"] is False
    assert all(len(runner["samples"]) == 10 for runner in report["runners"].values())


def test_output_change_or_missing_runner_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = {
        "allocated_bytes": 32 * 1024 * 1024,
        "worker_seconds": 0.025,
        "process_wall_seconds": 0.03,
        "process_peak_rss_mb": 40.0,
    }
    monkeypatch.setattr(performance, "run_calibration_process", lambda: calibration)
    monkeypatch.setattr(performance, "_run_process", lambda runner: _sample(runner))
    report = performance.collect_evidence(10)

    changed = copy.deepcopy(report)
    changed["runners"]["backtrader:ml4t"]["samples"][0]["output_sha256"] = "0" * 64
    assert performance.report_failures(changed)

    missing = copy.deepcopy(report)
    del missing["runners"]["zipline:external"]
    assert performance.report_failures(missing)


def test_worker_uses_profile_specific_ml4t_side(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = performance.RUNNERS_BY_ID["backtrader:ml4t"]
    captured: dict[str, object] = {}

    def benchmark(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(performance.suite, "benchmark_ml4t", benchmark)

    result = performance._run_ml4t(runner, object(), {}, object(), object())

    assert result is not None
    assert captured == {
        "execution_mode": "next_bar",
        "profile_override": "backtrader_strict",
    }


def test_accepted_write_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "accepted.json"
    target.write_text("old\n", encoding="utf-8")

    performance._write_json_atomic(target, {"value": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    assert list(tmp_path.iterdir()) == [target]
