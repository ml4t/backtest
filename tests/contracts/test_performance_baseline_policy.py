from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from validation import performance_baseline

ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "validation" / "performance_baselines.json"
WORKLOADS = {
    "single_asset",
    "daily_250_assets",
    "quote_aware",
    "rebalance",
    "partial_fill",
}


def test_release_performance_manifest_covers_required_workloads() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    performance_baseline._validate_manifest(manifest)
    assert manifest["schema_version"] == 2
    assert set(manifest["workloads"]) == WORKLOADS
    assert manifest["measurement_contract"] == {
        "runtime": "perf_counter around Engine.run only; setup excluded",
        "setup": "deterministic data, DataFeed, strategy, config, and Engine construction",
        "memory": "child-process peak RSS from interpreter start through completed run",
        "sample_deviation_reporting_threshold": 0.10,
        "regression_gate": "instrument-free hotpath benchmark in tests/benchmark",
    }
    for workload in manifest["workloads"].values():
        assert workload["behavior_sha256"]
        assert workload["data_points"] > 0


def test_public_docs_do_not_publish_unretained_performance_numbers() -> None:
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    text = "\n".join(path.read_text(encoding="utf-8") for path in documents)
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    retained_evidence = re.compile(r"performance_baselines\.json|release-performance-evidence")
    ratio_or_throughput = re.compile(
        r"\b\d+(?:\.\d+)?x\s+(?:faster|slower|less)|"
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:bars/s|rows/s|points/s)|"
        r"\b\d[\d,]*(?:\.\d+)?\s+(?:bars|rows|points)\s+per\s+second",
        re.IGNORECASE,
    )
    resource_claim = re.compile(r"\b\d+(?:\.\d+)?\s*(?:MB|GB|seconds?)\b", re.IGNORECASE)
    resource_context = re.compile(
        r"\b(?:benchmark|memory|performance|rss|runtime)\b", re.IGNORECASE
    )
    violations = [
        line
        for line in prose.splitlines()
        if not retained_evidence.search(line)
        and (
            ratio_or_throughput.search(line)
            or resource_claim.search(line)
            and resource_context.search(line)
        )
    ]

    assert not violations


def test_runtime_sample_spread_is_reported_without_becoming_a_host_noise_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["workloads"] = {"single_asset": manifest["workloads"]["single_asset"]}
    monkeypatch.setattr(
        performance_baseline,
        "WORKLOADS",
        {"single_asset": performance_baseline.WORKLOADS["single_asset"]},
    )
    expected = manifest["workloads"]["single_asset"]
    samples = iter(
        {
            "data_points": expected["data_points"],
            "fill_count": expected["expected_fill_count"],
            "trade_count": expected["expected_trade_count"],
            "final_value": expected["expected_final_value"],
            "behavior_sha256": expected["behavior_sha256"],
            "runtime_seconds": runtime,
            "setup_seconds": 0.1,
            "total_measured_seconds": runtime + 0.1,
            "process_peak_rss_mb": memory,
        }
        for runtime, memory in [(1.0, 100.0), (1.1, 101.0), (3.0, 180.0)]
    )
    monkeypatch.setattr(performance_baseline, "_worker_sample", lambda _name: next(samples))

    evidence = performance_baseline.collect_evidence(manifest, samples=3)

    workload = evidence["workloads"]["single_asset"]
    assert workload["passed"] is True
    assert workload["sample_spread_within_reporting_threshold"] is False
    assert workload["behavior_sha256"] == [expected["behavior_sha256"]]


def test_worker_failure_reports_captured_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(
        args=["python"], returncode=3, stdout="", stderr="worker exploded"
    )
    monkeypatch.setattr(
        performance_baseline.subprocess,
        "run",
        lambda *_args, **_kwargs: completed,
    )

    with pytest.raises(RuntimeError, match="exited 3: worker exploded"):
        performance_baseline._worker_sample("single_asset")
