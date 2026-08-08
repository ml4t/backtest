from __future__ import annotations

import json
from pathlib import Path

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

    assert manifest["schema_version"] == 1
    assert set(manifest["workloads"]) == WORKLOADS
    assert manifest["measurement_contract"] == {
        "runtime": "perf_counter around Engine.run only; setup excluded",
        "setup": "deterministic data, DataFeed, strategy, config, and Engine construction",
        "memory": "child-process peak RSS from interpreter start through completed run",
        "reproducibility_tolerance": 0.10,
    }
    for workload in manifest["workloads"].values():
        assert workload["behavior_sha256"]
        assert workload["data_points"] > 0


def test_public_docs_do_not_publish_unretained_performance_numbers() -> None:
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    text = "\n".join(path.read_text(encoding="utf-8") for path in documents)

    assert "For 1M bars" not in text
    assert "10x less" not in text
