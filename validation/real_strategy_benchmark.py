#!/usr/bin/env python3
"""Measure engine-only runtimes for correctness-passing real-strategy pairs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.provenance import _tree_digest
from real_strategy_evidence import ADAPTER_PATHS, _sha256
from real_strategy_evidence import (
    report_failures as real_strategy_report_failures,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "validation"
APPLICABILITY = tomllib.loads(
    (VALIDATION_DIR / "real_strategy_applicability.toml").read_text(encoding="utf-8")
)
BUNDLES = APPLICABILITY["bundle"]
TIMING_SOURCE_PATHS = {
    **{name: path for name, path in ADAPTER_PATHS.items() if name != "evidence_builder"},
    "benchmark": Path(__file__).resolve(),
}
PYTHONS = {
    "ml4t": ".venv/bin/python",
    "vectorbt_pro": ".venv-vectorbt-pro/bin/python",
    "vectorbt_oss": ".venv-vectorbt-oss/bin/python",
    "backtrader": ".venv-backtrader/bin/python",
    "zipline": ".venv-zipline/bin/python",
    "lean": ".venv/bin/python",
}
PROFILES = {
    ("etfs", "vectorbt_pro"): ("vectorbt_strict", []),
    ("etfs", "vectorbt_oss"): ("vectorbt_oss_strict", []),
    ("etfs", "backtrader"): ("backtrader_strict", []),
    ("etfs", "zipline"): ("zipline_strict", ["--price-decimals", "3"]),
    ("etfs", "lean"): ("lean", ["--price-decimals", "4"]),
    ("cme_futures", "vectorbt_pro"): ("vectorbt_futures_strict", []),
    ("cme_futures", "backtrader"): ("backtrader_strict", []),
    ("fx_pairs", "vectorbt_pro"): ("vectorbt_strict", []),
    ("fx_pairs", "vectorbt_oss"): ("vectorbt_oss_strict", []),
    ("fx_pairs", "backtrader"): ("backtrader_strict", []),
    ("fx_pairs", "lean"): ("lean", ["--price-decimals", "5"]),
    (
        "crypto_perps_funding",
        "lean",
    ): (
        "lean_crypto_future",
        [
            "--execution-specs",
            "validation/lean/workspace/real_strategy_crypto_perps_funding/symbol_properties.json",
        ],
    ),
}
ADAPTERS = {
    "vectorbt_pro": ("real_strategy_vectorbt.py", ["--framework", "vectorbt_pro"]),
    "vectorbt_oss": ("real_strategy_vectorbt.py", ["--framework", "vectorbt_oss"]),
    "backtrader": ("real_strategy_backtrader.py", []),
    "zipline": ("real_strategy_zipline.py", []),
    "lean": ("real_strategy_lean.py", []),
}


def bootstrap_median_interval(
    values: list[float], *, draws: int = 10_000, seed: int = 20260814
) -> tuple[float, float]:
    """Return the percentile bootstrap 95 percent interval for a sample median."""
    if not values:
        raise ValueError("Cannot bootstrap an empty timing sample")
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values))) for _ in range(draws)
    )
    return medians[int(0.025 * draws)], medians[int(0.975 * draws)]


def _command(
    *, side: str, case_study: str, framework: str, bundle: Path, output: Path
) -> list[str]:
    python = PROJECT_ROOT / PYTHONS[framework if side == "framework" else "ml4t"]
    if side == "framework":
        script, extra = ADAPTERS[framework]
        if case_study == "crypto_perps_funding":
            script = "real_strategy_lean_crypto.py"
        elif case_study == "fx_pairs" and framework == "lean":
            script = "real_strategy_lean_fx.py"
        return [
            str(python),
            str(VALIDATION_DIR / script),
            "--bundle",
            str(bundle),
            "--output",
            str(output),
            *extra,
        ]
    profile, extra = PROFILES[(case_study, framework)]
    return [
        str(python),
        str(VALIDATION_DIR / "real_strategy_runner.py"),
        "--bundle",
        str(bundle),
        "--output",
        str(output),
        "--comparison-profile",
        profile,
        *extra,
    ]


def _output_identity(manifest: dict[str, Any]) -> dict[str, str]:
    retained = {"fills.parquet", "equity.parquet", "rejected_orders.parquet"}
    return {
        name: identity["sha256"] for name, identity in manifest["files"].items() if name in retained
    }


def _run_once(command: list[str], output: Path) -> tuple[float, dict[str, str]]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Benchmark subprocess failed: {detail[-1000:]}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    return float(manifest["engine_seconds"]), _output_identity(manifest)


def _measure_side(
    *,
    side: str,
    case_study: str,
    framework: str,
    bundle: Path,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    timings: list[float] = []
    identity: dict[str, str] | None = None
    with tempfile.TemporaryDirectory(prefix="ml4t-real-benchmark-") as temporary:
        temporary_root = Path(temporary)
        for index in range(warmups + samples):
            output = temporary_root / f"run-{index}"
            command = _command(
                side=side,
                case_study=case_study,
                framework=framework,
                bundle=bundle,
                output=output,
            )
            runtime, current_identity = _run_once(command, output)
            if identity is None:
                identity = current_identity
            elif current_identity != identity:
                raise RuntimeError(
                    f"Output changed across repetitions for {case_study}/{framework}/{side}"
                )
            if index >= warmups:
                timings.append(runtime)
    lower, upper = bootstrap_median_interval(timings)
    return {
        "samples_seconds": timings,
        "median_seconds": statistics.median(timings),
        "ci_95_seconds": [lower, upper],
        "minimum_seconds": min(timings),
        "maximum_seconds": max(timings),
        "output_identity": identity,
    }


def build_report(
    *,
    correctness: dict[str, Any],
    bundle_root: Path,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    """Measure every pair that passed the supplied correctness evidence."""
    passing = [record for record in correctness["records"] if record["status"] == "pass"]
    records: list[dict[str, object]] = []
    for index, pair in enumerate(passing, start=1):
        case_study = pair["case_study"]
        framework = pair["framework"]
        print(f"Timing {index}/{len(passing)} {case_study}/{framework}", flush=True)
        bundle = bundle_root / case_study / BUNDLES[case_study]
        framework_result = _measure_side(
            side="framework",
            case_study=case_study,
            framework=framework,
            bundle=bundle,
            warmups=warmups,
            samples=samples,
        )
        ml4t_result = _measure_side(
            side="ml4t",
            case_study=case_study,
            framework=framework,
            bundle=bundle,
            warmups=warmups,
            samples=samples,
        )
        records.append(
            {
                "case_study": case_study,
                "framework": framework,
                "input_bundle_sha256": pair["input_bundle_sha256"],
                "correctness_status": "pass",
                "framework_engine": framework_result,
                "ml4t_engine": ml4t_result,
                "framework_to_ml4t_median_ratio": (
                    framework_result["median_seconds"] / ml4t_result["median_seconds"]
                ),
            }
        )
    targets = tomllib.loads((VALIDATION_DIR / "framework_targets.toml").read_text())
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "timing_policy": {
            "boundary": "engine call only",
            "excluded": [
                "input loading",
                "model inference",
                "target construction",
                "adapter preparation",
                "result extraction",
                "serialization",
                "reporting",
            ],
            "warmup_processes": warmups,
            "measured_processes": samples,
            "process_isolation": True,
            "interval": "10,000-draw percentile bootstrap of the sample median",
            "publication_scope": "only correctness-passing case-study/framework pairs",
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "frameworks": targets["framework"],
        },
        "provenance": {
            "ml4t_engine_source_sha256": _tree_digest(PROJECT_ROOT / "src/ml4t/backtest"),
            "sources": {name: _sha256(path) for name, path in TIMING_SOURCE_PATHS.items()},
        },
        "correctness_evidence_generated_at": correctness["generated_at"],
        "records": records,
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def report_failures(report: dict[str, Any], correctness: dict[str, Any]) -> list[str]:
    """Return every reason real-strategy timing evidence is not publication-safe."""
    failures: list[str] = []
    if report.get("schema_version") != 1:
        return [f"Unsupported real-strategy performance schema: {report.get('schema_version')!r}"]
    if not isinstance(report.get("generated_at"), str):
        failures.append("Real-strategy performance report lacks a generation timestamp")
    if report.get("correctness_evidence_generated_at") != correctness.get("generated_at"):
        failures.append("Real-strategy performance evidence references stale correctness evidence")

    timing = report.get("timing_policy")
    if not isinstance(timing, dict):
        failures.append("Real-strategy timing policy must be an object")
        measured_processes = 0
    else:
        expected_static_policy = {
            "boundary": "engine call only",
            "excluded": [
                "input loading",
                "model inference",
                "target construction",
                "adapter preparation",
                "result extraction",
                "serialization",
                "reporting",
            ],
            "process_isolation": True,
            "interval": "10,000-draw percentile bootstrap of the sample median",
            "publication_scope": "only correctness-passing case-study/framework pairs",
        }
        if any(timing.get(key) != value for key, value in expected_static_policy.items()):
            failures.append("Real-strategy timing policy differs")
        warmups = timing.get("warmup_processes")
        measured_processes = timing.get("measured_processes")
        if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 1:
            failures.append("Real-strategy timings lack a warm-up process")
        if (
            not isinstance(measured_processes, int)
            or isinstance(measured_processes, bool)
            or measured_processes < 10
        ):
            failures.append("Real-strategy timings require at least ten measured processes")
            measured_processes = 0

    targets = tomllib.loads((VALIDATION_DIR / "framework_targets.toml").read_text())
    environment = report.get("environment")
    if not isinstance(environment, dict):
        failures.append("Real-strategy performance environment must be an object")
    else:
        if environment.get("frameworks") != targets["framework"]:
            failures.append("Real-strategy performance framework targets differ")
        platform_name = environment.get("platform")
        if not isinstance(platform_name, str) or not platform_name.startswith("Linux-"):
            failures.append("Real-strategy publication timings must identify a Linux host")
        for field in ("machine", "processor"):
            if not isinstance(environment.get(field), str) or not environment[field]:
                failures.append(f"Real-strategy performance environment lacks {field}")
        cpu_count = environment.get("cpu_count")
        if not isinstance(cpu_count, int) or isinstance(cpu_count, bool) or cpu_count < 1:
            failures.append("Real-strategy performance environment lacks a valid CPU count")

    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        failures.append("Real-strategy performance provenance must be an object")
    else:
        if provenance.get("ml4t_engine_source_sha256") != _tree_digest(
            PROJECT_ROOT / "src/ml4t/backtest"
        ):
            failures.append("Real-strategy performance engine source digest is stale")
        expected_sources = {name: _sha256(path) for name, path in TIMING_SOURCE_PATHS.items()}
        if provenance.get("sources") != expected_sources:
            failures.append("Real-strategy performance runner source digests are stale")

    correctness_records = correctness.get("records")
    if not isinstance(correctness_records, list):
        failures.append("Real-strategy correctness records must be a list")
        correctness_records = []
    expected_records = {
        (record["case_study"], record["framework"]): record
        for record in correctness_records
        if isinstance(record, dict) and record.get("status") == "pass"
    }
    records = report.get("records")
    if not isinstance(records, list):
        failures.append("Real-strategy performance records must be a list")
        records = []
    records_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"Real-strategy performance record {index} must be an object")
            continue
        pair = (record.get("case_study"), record.get("framework"))
        if not all(isinstance(value, str) for value in pair):
            failures.append(f"Real-strategy performance record {index} lacks a pair identity")
            continue
        identity = (str(pair[0]), str(pair[1]))
        if identity in records_by_pair:
            failures.append(f"Real-strategy performance duplicates {identity[0]}/{identity[1]}")
        records_by_pair[identity] = record
    if set(records_by_pair) != set(expected_records):
        failures.append("Real-strategy performance pairs differ from correctness-passing pairs")

    for pair in sorted(set(records_by_pair) & set(expected_records)):
        case_study, framework = pair
        identity = f"{case_study}/{framework}"
        record = records_by_pair[pair]
        correctness_record = expected_records[pair]
        if record.get("correctness_status") != "pass":
            failures.append(f"{identity} timing record is not tied to a passing comparison")
        if record.get("input_bundle_sha256") != correctness_record.get("input_bundle_sha256"):
            failures.append(f"{identity} timing input differs from correctness evidence")
        medians: dict[str, float] = {}
        for side in ("framework_engine", "ml4t_engine"):
            result = record.get(side)
            if not isinstance(result, dict):
                failures.append(f"{identity} lacks {side} timings")
                continue
            samples = result.get("samples_seconds")
            if (
                not isinstance(samples, list)
                or len(samples) != measured_processes
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value > 0
                    for value in samples
                )
            ):
                failures.append(f"{identity}/{side} lacks valid raw timing samples")
                continue
            numeric_samples = [float(value) for value in samples]
            median = statistics.median(numeric_samples)
            medians[side] = median
            if result.get("median_seconds") != median:
                failures.append(f"{identity}/{side} median differs from raw samples")
            if result.get("minimum_seconds") != min(numeric_samples):
                failures.append(f"{identity}/{side} minimum differs from raw samples")
            if result.get("maximum_seconds") != max(numeric_samples):
                failures.append(f"{identity}/{side} maximum differs from raw samples")
            if result.get("ci_95_seconds") != list(bootstrap_median_interval(numeric_samples)):
                failures.append(f"{identity}/{side} interval differs from raw samples")
            output_identity = result.get("output_identity")
            if (
                not isinstance(output_identity, dict)
                or not {"fills.parquet", "equity.parquet"} <= set(output_identity)
                or not all(_is_sha256(value) for value in output_identity.values())
            ):
                failures.append(f"{identity}/{side} lacks deterministic output identities")
        if set(medians) == {"framework_engine", "ml4t_engine"}:
            expected_ratio = medians["framework_engine"] / medians["ml4t_engine"]
            if record.get("framework_to_ml4t_median_ratio") != expected_ratio:
                failures.append(f"{identity} timing ratio differs from raw samples")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness",
        type=Path,
        default=VALIDATION_DIR / "REAL_STRATEGY_RESULTS.json",
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATION_DIR / "candidates/REAL_STRATEGY_PERFORMANCE.candidate.json",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 10:
        raise ValueError("Benchmarks require at least one warmup and ten measured processes")
    correctness = json.loads(args.correctness.read_text(encoding="utf-8"))
    correctness_failures = real_strategy_report_failures(correctness)
    if correctness_failures:
        raise ValueError(
            "Real-strategy correctness evidence is invalid: " + "; ".join(correctness_failures)
        )
    report = build_report(
        correctness=correctness,
        bundle_root=args.bundle_root.resolve(),
        warmups=args.warmups,
        samples=args.samples,
    )
    failures = report_failures(report, correctness)
    if failures:
        raise ValueError("Real-strategy performance evidence is invalid: " + "; ".join(failures))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Retained {len(report['records'])} timing comparisons: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
