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
SUBPROCESS_TIMEOUT_SECONDS = 900
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
    ("us_equities_panel", "vectorbt_pro"): ("vectorbt_strict", []),
    ("us_equities_panel", "vectorbt_oss"): ("vectorbt_oss_strict", []),
    ("us_equities_panel", "backtrader"): ("backtrader_strict", []),
    ("us_equities_panel", "zipline"): ("zipline_strict", ["--price-decimals", "3"]),
    ("us_equities_panel", "lean"): ("lean", ["--price-decimals", "4"]),
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
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
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


def certified_sources(provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the engine sources these timings are certified for, keyed by digest."""
    entries = provenance.get("certified_equivalent_sources")
    if not isinstance(entries, list):
        return {}
    return {
        entry["engine_source_sha256"]: entry
        for entry in entries
        if isinstance(entry, dict) and _is_sha256(entry.get("engine_source_sha256"))
    }


def source_is_published(provenance: dict[str, Any], current: str) -> bool:
    """Whether the current engine source is one these timings may be published for.

    The timings are measured under exactly one source tree. That digest lives in
    `ml4t_engine_source_sha256` and is never rewritten, so the report always says
    which source produced the numbers. A later source that moves no measured
    quantity is added to `certified_equivalent_sources` instead of triggering a
    re-measurement, because re-measuring costs 2h29m of exclusive machine time and
    cannot change a value that correctness parity has already shown is identical.

    `_tree_digest` hashes every tracked `.py` under the engine, so a docstring edit
    moves it exactly as far as a rewritten fill model does. Without certification
    that granularity prices a full benchmark run for any source edit at all.
    """
    return current == provenance.get("ml4t_engine_source_sha256") or current in certified_sources(
        provenance
    )


def runners_are_published(provenance: dict[str, Any], current: dict[str, str]) -> bool:
    """Whether the current timing runners are ones these timings may be published for.

    Same contract as `source_is_published`, for the adapter and benchmark scripts
    rather than the engine. Editing this file to add a certification path changes
    its own digest without touching a single measured number, which is the defect
    this function exists to stop recurring one level up from the engine.
    """
    if current == provenance.get("sources"):
        return True
    return any(entry.get("sources") == current for entry in certified_sources(provenance).values())


def certification_shape_failures(provenance: dict[str, Any]) -> list[str]:
    """Return every reason a certification entry cannot be trusted."""
    entries = provenance.get("certified_equivalent_sources")
    if entries is None:
        return []
    if not isinstance(entries, list):
        return ["Real-strategy certified_equivalent_sources must be a list"]
    failures: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"Certification {index} is not an object")
            continue
        if not _is_sha256(entry.get("engine_source_sha256")):
            failures.append(f"Certification {index} lacks a valid engine source digest")
        if not _is_sha256(entry.get("correctness_evidence_sha256")):
            failures.append(f"Certification {index} lacks the correctness evidence it rests on")
        for field in ("certified_at", "reason", "ml4t_commit"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                failures.append(f"Certification {index} lacks {field}")
        sources = entry.get("sources")
        if not isinstance(sources, dict) or not all(
            _is_sha256(value) for value in sources.values()
        ):
            failures.append(f"Certification {index} lacks valid runner source digests")
        passing = entry.get("correctness_pairs_passed")
        if not isinstance(passing, int) or isinstance(passing, bool) or passing < 1:
            failures.append(f"Certification {index} lacks a passing-pair count")
    return failures


def certify_source(
    report: dict[str, Any],
    correctness: dict[str, Any],
    correctness_path: Path,
    reason: str,
) -> dict[str, Any]:
    """Certify the current engine source against timings measured under an older one.

    This never invents a measurement. It records that correctness parity was
    re-derived under the current source and moved no value, so the published
    timings stay attributed to the source that produced them while remaining
    publishable for this one.

    Correctness parity does not by itself prove a source change is performance
    inert - an added loop can leave every number identical and still cost time.
    That judgement is the caller's, which is why `reason` is required and stored.
    """
    current = _tree_digest(PROJECT_ROOT / "src/ml4t/backtest")
    provenance = report["provenance"]
    measured_under = provenance.get("ml4t_engine_source_sha256")
    if current == measured_under:
        raise ValueError("Engine source is unchanged; there is nothing to certify")
    if current in certified_sources(provenance):
        raise ValueError(f"Engine source {current} is already certified")
    if not reason.strip():
        raise ValueError("Certification requires a reason the change cannot move a timing")

    correctness_failures = real_strategy_report_failures(correctness)
    if correctness_failures:
        raise ValueError("Correctness evidence is invalid: " + "; ".join(correctness_failures))
    correctness_digest = correctness["provenance"]["ml4t"]["engine_source_sha256"]
    if correctness_digest != current:
        raise ValueError(
            f"Correctness evidence was produced under engine source {correctness_digest}, "
            f"not the working tree's {current}; re-derive correctness before certifying"
        )
    passing = [record for record in correctness["records"] if record.get("status") == "pass"]
    if not passing:
        raise ValueError("Correctness evidence contains no passing pair")

    entry = {
        "engine_source_sha256": current,
        "certified_at": datetime.now(UTC).isoformat(),
        "correctness_evidence_sha256": _sha256(correctness_path),
        "correctness_evidence_generated_at": correctness["generated_at"],
        "correctness_pairs_passed": len(passing),
        "ml4t_commit": correctness["provenance"]["ml4t"]["commit"],
        "reason": reason.strip(),
        "sources": {name: _sha256(path) for name, path in TIMING_SOURCE_PATHS.items()},
    }
    provenance.setdefault("certified_equivalent_sources", []).append(entry)
    # The timings are now attested by this correctness run rather than the one they
    # were measured beside, so the pointer has to move with it or report_failures
    # reads the evidence as stale. The measured-under digest above does not move.
    report["correctness_evidence_generated_at"] = correctness["generated_at"]
    return entry


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
        certification_failures = certification_shape_failures(provenance)
        failures.extend(certification_failures)
        if not certification_failures and not source_is_published(
            provenance, _tree_digest(PROJECT_ROOT / "src/ml4t/backtest")
        ):
            failures.append(
                "Real-strategy performance engine source is neither the source these "
                "timings were measured under nor one they are certified for"
            )
        expected_sources = {name: _sha256(path) for name, path in TIMING_SOURCE_PATHS.items()}
        if not certification_failures and not runners_are_published(provenance, expected_sources):
            failures.append(
                "Real-strategy performance runner sources are neither the runners these "
                "timings were measured with nor ones they are certified for"
            )

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
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument(
        "--performance",
        type=Path,
        default=VALIDATION_DIR / "REAL_STRATEGY_PERFORMANCE.json",
        help="Published timing evidence to certify a new engine source against.",
    )
    parser.add_argument(
        "--certify-source",
        action="store_true",
        help=(
            "Do not benchmark. Record that the current engine source moves no measured "
            "value, so the published timings stay valid for it. Requires --reason and "
            "correctness evidence re-derived under the current source."
        ),
    )
    parser.add_argument(
        "--reason",
        default="",
        help="Why the source change cannot move a timing. Stored in the certification.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATION_DIR / "candidates/REAL_STRATEGY_PERFORMANCE.candidate.json",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.certify_source:
        correctness = json.loads(args.correctness.read_text(encoding="utf-8"))
        report = json.loads(args.performance.read_text(encoding="utf-8"))
        entry = certify_source(report, correctness, args.correctness, args.reason)
        failures = report_failures(report, correctness)
        if failures:
            raise ValueError("Certified performance evidence is invalid: " + "; ".join(failures))
        args.performance.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"Certified engine source {entry['engine_source_sha256'][:12]} against timings "
            f"measured under {report['provenance']['ml4t_engine_source_sha256'][:12]}: "
            f"{entry['correctness_pairs_passed']} correctness pairs pass, no timing re-measured."
        )
        return 0
    if args.bundle_root is None:
        raise ValueError("--bundle-root is required unless --certify-source is given")
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
