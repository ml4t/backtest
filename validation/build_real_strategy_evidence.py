#!/usr/bin/env python3
"""Produce the retained real-strategy output tree that the comparator reads.

``real_strategy_evidence.py`` compares ``<root>/<case_study>/<framework>`` against
``<root>/<case_study>/ml4t_<framework>`` but does not produce either, and nothing else in
``validation/`` did: the one place that invokes every side correctly is
``real_strategy_benchmark._measure_side``, which writes into a ``TemporaryDirectory`` because
it only wants the timings. This runs the same invocations once per side into a durable tree,
so the evidence can be rebuilt after an engine source change instead of by hand.

    uv run python validation/build_real_strategy_evidence.py \
      --bundle-root PATH/TO/CONTENT_ADDRESSED_BUNDLES \
      --evidence-root PATH/TO/REAL_STRATEGY_OUTPUTS

Timings are not collected here. One run per side is enough for the comparator and is the wrong
sample for a published measurement; ``real_strategy_benchmark.py`` remains the only thing that
reports engine time, and it needs an unloaded machine.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from real_strategy_benchmark import (  # noqa: E402
    BUNDLES,
    PROJECT_ROOT,
    VALIDATION_DIR as _BENCHMARK_VALIDATION_DIR,
    _command,
    _run_once,
)

APPLICABILITY_PATH = VALIDATION_DIR / "real_strategy_applicability.toml"
assert _BENCHMARK_VALIDATION_DIR == VALIDATION_DIR


def required_pairs() -> list[tuple[str, str]]:
    applicability = tomllib.loads(APPLICABILITY_PATH.read_text(encoding="utf-8"))
    return [
        (pair["case_study"], pair["framework"])
        for pair in applicability["pair"]
        if pair["status"] == "required"
    ]


def build(*, bundle_root: Path, evidence_root: Path) -> int:
    pairs = required_pairs()
    for index, (case_study, framework) in enumerate(pairs, start=1):
        bundle = bundle_root / case_study / BUNDLES[case_study]
        if not bundle.is_dir():
            raise FileNotFoundError(f"Bundle missing for {case_study}: {bundle}")
        for side, directory in (
            ("framework", evidence_root / case_study / framework),
            ("ml4t", evidence_root / case_study / f"ml4t_{framework}"),
        ):
            directory.parent.mkdir(parents=True, exist_ok=True)
            command = _command(
                side=side,
                case_study=case_study,
                framework=framework,
                bundle=bundle,
                output=directory,
            )
            engine_seconds, identity = _run_once(command, directory)
            print(
                f"{index}/{len(pairs)} {case_study}/{framework}/{side}: "
                f"{engine_seconds:.6f}s engine time -> {directory}",
                flush=True,
            )
            if "equity.parquet" not in identity or "fills.parquet" not in identity:
                raise RuntimeError(
                    f"{case_study}/{framework}/{side} retained {sorted(identity)}; the "
                    "comparator needs equity.parquet and fills.parquet"
                )
    print(f"Wrote {2 * len(pairs)} sides for {len(pairs)} pairs under {evidence_root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    return build(
        bundle_root=args.bundle_root.resolve(),
        evidence_root=args.evidence_root.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
