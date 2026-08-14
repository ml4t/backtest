#!/usr/bin/env python3
"""Measure engine-only runtimes for correctness-passing real-strategy pairs."""

from __future__ import annotations

import argparse
import json
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "validation"
BUNDLES = {
    "etfs": "01f38079ce47821a5379d3769e86f4a2170b88033108153bfc0f928698e946db",
    "cme_futures": "c7191027f550ef1be0c9528dac08e7c2bf6a76c86d935f593efeb1ad8b628c39",
    "crypto_perps_funding": "2acee3c8542043266e6f9c0dfc9434c95b4469ef6fcaf0556efc569fc9721cdd",
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
    ("cme_futures", "vectorbt_pro"): ("vectorbt_strict", []),
    ("cme_futures", "backtrader"): ("backtrader_strict", []),
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
        "correctness_evidence_generated_at": correctness["generated_at"],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness",
        type=Path,
        default=VALIDATION_DIR / "candidates/REAL_STRATEGY_RESULTS.candidate.json",
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
    report = build_report(
        correctness=correctness,
        bundle_root=args.bundle_root.resolve(),
        warmups=args.warmups,
        samples=args.samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Retained {len(report['records'])} timing comparisons: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
