#!/usr/bin/env python3
"""Build retained exact-comparison evidence for frozen real strategies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "validation"
CANONICAL_QUANTUM = Decimal("0.00000001")
PAIR_PROFILES = {
    ("etfs", "vectorbt_pro"): "vectorbt_strict",
    ("etfs", "vectorbt_oss"): "vectorbt_oss_strict",
    ("etfs", "backtrader"): "backtrader_strict",
    ("etfs", "zipline"): "zipline_strict",
    ("etfs", "lean"): "lean",
    ("cme_futures", "vectorbt_pro"): "vectorbt_strict",
    ("cme_futures", "backtrader"): "backtrader_strict",
    ("crypto_perps_funding", "lean"): "lean_crypto_future",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _number_text(value: object) -> str:
    return format(Decimal(str(value)), "f")


def _canonical_gap(framework: str, ml4t: str) -> Decimal:
    return abs(Decimal(framework) - Decimal(ml4t)).quantize(CANONICAL_QUANTUM, ROUND_HALF_EVEN)


def _canonical_timestamp(value: object, *, intraday: bool) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds") if intraday else value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.isoformat(timespec="seconds") if intraday else parsed.date().isoformat()


def _fill_records(frame: pl.DataFrame, *, intraday: bool) -> list[dict[str, str]]:
    records = [
        {
            "timestamp": _canonical_timestamp(row["timestamp"], intraday=intraday),
            "asset": str(row["asset"]),
            "side": str(row["side"]).lower(),
            "quantity": _number_text(row["quantity"]),
            "price": _number_text(row["price"]),
            "commission": _number_text(row["commission"]),
        }
        for row in frame.iter_rows(named=True)
    ]
    return sorted(
        records,
        key=lambda row: (
            row["timestamp"],
            row["asset"],
            row["side"],
            row["quantity"],
            row["price"],
            row["commission"],
        ),
    )


def _value_records(frame: pl.DataFrame, *, field: str, intraday: bool) -> list[dict[str, str]]:
    records = [
        {
            "timestamp": _canonical_timestamp(row["timestamp"], intraday=intraday),
            field: _number_text(row[field]),
        }
        for row in frame.select("timestamp", field).iter_rows(named=True)
    ]
    return sorted(records, key=lambda row: row["timestamp"])


def _first_divergence(
    framework_records: list[dict[str, str]],
    ml4t_records: list[dict[str, str]],
    *,
    numeric_fields: tuple[str, ...],
) -> dict[str, object] | None:
    if len(framework_records) != len(ml4t_records):
        return {
            "kind": "record_count",
            "framework": len(framework_records),
            "ml4t": len(ml4t_records),
        }
    for index, (framework, ml4t) in enumerate(zip(framework_records, ml4t_records, strict=True)):
        fields = sorted(set(framework) | set(ml4t))
        for field in fields:
            framework_value = framework.get(field)
            ml4t_value = ml4t.get(field)
            if field in numeric_fields and framework_value is not None and ml4t_value is not None:
                differs = _canonical_gap(framework_value, ml4t_value) != 0
            else:
                differs = framework_value != ml4t_value
            if differs:
                return {
                    "kind": "record",
                    "index": index,
                    "field": field,
                    "framework": framework,
                    "ml4t": ml4t,
                    "canonical_gap": (
                        format(_canonical_gap(framework_value, ml4t_value), "f")
                        if field in numeric_fields
                        and framework_value is not None
                        and ml4t_value is not None
                        else None
                    ),
                }
    return None


def compare_records(
    framework_records: list[dict[str, str]],
    ml4t_records: list[dict[str, str]],
    *,
    numeric_fields: tuple[str, ...] = (),
) -> dict[str, object]:
    """Compare complete canonical streams and retain the first divergence."""
    divergence = _first_divergence(framework_records, ml4t_records, numeric_fields=numeric_fields)
    return {
        "passed": divergence is None,
        "framework_records": len(framework_records),
        "ml4t_records": len(ml4t_records),
        "framework_sha256": _json_digest(framework_records),
        "ml4t_sha256": _json_digest(ml4t_records),
        "first_divergence": divergence,
    }


def _max_difference(
    framework_records: list[dict[str, str]],
    ml4t_records: list[dict[str, str]],
    *,
    fields: tuple[str, ...],
) -> str | None:
    if len(framework_records) != len(ml4t_records):
        return None
    differences: list[Decimal] = []
    for framework, ml4t in zip(framework_records, ml4t_records, strict=True):
        for field in fields:
            if field in framework and field in ml4t:
                differences.append(abs(Decimal(framework[field]) - Decimal(ml4t[field])))
    maximum = max(differences, default=Decimal(0))
    return format(maximum.quantize(CANONICAL_QUANTUM, ROUND_HALF_EVEN), "f")


def _align_shared_values(
    framework_records: list[dict[str, str]], ml4t_records: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int | str | None]]:
    framework_by_time = {record["timestamp"]: record for record in framework_records}
    ml4t_by_time = {record["timestamp"]: record for record in ml4t_records}
    shared = sorted(framework_by_time.keys() & ml4t_by_time.keys())
    framework_only = sorted(framework_by_time.keys() - ml4t_by_time.keys())
    ml4t_only = sorted(ml4t_by_time.keys() - framework_by_time.keys())
    return (
        [framework_by_time[timestamp] for timestamp in shared],
        [ml4t_by_time[timestamp] for timestamp in shared],
        {
            "shared_timestamps": len(shared),
            "framework_only_timestamps": len(framework_only),
            "ml4t_only_timestamps": len(ml4t_only),
            "first_framework_only_timestamp": framework_only[0] if framework_only else None,
            "first_ml4t_only_timestamp": ml4t_only[0] if ml4t_only else None,
        },
    )


def _value_surface(
    framework_records: list[dict[str, str]],
    ml4t_records: list[dict[str, str]],
    *,
    field: str,
) -> tuple[dict[str, object], list[dict[str, str]], list[dict[str, str]]]:
    framework_shared, ml4t_shared, coverage = _align_shared_values(framework_records, ml4t_records)
    if not framework_shared:
        raise ValueError("No shared valuation timestamps")
    surface = _surface(framework_shared, ml4t_shared, numeric_fields=(field,))
    coverage_passed = not (
        coverage["framework_only_timestamps"] or coverage["ml4t_only_timestamps"]
    )
    surface["coverage"] = coverage
    surface["coverage_passed"] = coverage_passed
    if not coverage_passed:
        surface["passed"] = False
        if surface["first_divergence"] is None:
            surface["first_divergence"] = {
                "kind": "timestamp_coverage",
                "framework_only": coverage["first_framework_only_timestamp"],
                "ml4t_only": coverage["first_ml4t_only_timestamp"],
            }
    return surface, framework_shared, ml4t_shared


def _load_evidence(path: Path) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    frames: dict[str, pl.DataFrame] = {}
    for name, identity in manifest["files"].items():
        artifact = path / name
        if not artifact.is_file() or _sha256(artifact) != identity["sha256"]:
            raise ValueError(f"Evidence identity mismatch: {artifact}")
        if artifact.suffix == ".parquet":
            frames[artifact.stem] = pl.read_parquet(artifact)
    return manifest, frames


def _surface(
    framework_records: list[dict[str, str]],
    ml4t_records: list[dict[str, str]],
    *,
    numeric_fields: tuple[str, ...],
) -> dict[str, object]:
    result = compare_records(framework_records, ml4t_records, numeric_fields=numeric_fields)
    result["max_canonical_difference"] = _max_difference(
        framework_records, ml4t_records, fields=numeric_fields
    )
    return result


def _comparison_record(
    *, case_study: str, framework: str, evidence_root: Path
) -> dict[str, object]:
    external_path = evidence_root / case_study / framework
    ml4t_path = evidence_root / case_study / f"ml4t_{framework}"
    external, external_frames = _load_evidence(external_path)
    ml4t, ml4t_frames = _load_evidence(ml4t_path)
    expected_profile = PAIR_PROFILES[(case_study, framework)]
    if external["input_bundle_sha256"] != ml4t["input_bundle_sha256"]:
        raise ValueError(f"Input bundle differs for {case_study}/{framework}")
    if external["comparison_profile"] != expected_profile:
        raise ValueError(f"External profile differs for {case_study}/{framework}")
    if ml4t["comparison_profile"] != expected_profile:
        raise ValueError(f"ML4T profile differs for {case_study}/{framework}")

    intraday = case_study == "crypto_perps_funding"
    external_fills = _fill_records(external_frames["fills"], intraday=intraday)
    ml4t_fills = _fill_records(ml4t_frames["fills"], intraday=intraday)
    external_equity = _value_records(external_frames["equity"], field="equity", intraday=intraday)
    ml4t_equity = _value_records(ml4t_frames["equity"], field="equity", intraday=intraday)
    equity_surface, external_equity, ml4t_equity = _value_surface(
        external_equity, ml4t_equity, field="equity"
    )
    terminal_external = [{"final_value": external_equity[-1]["equity"]}]
    terminal_ml4t = [{"final_value": ml4t_equity[-1]["equity"]}]
    surfaces: dict[str, dict[str, object]] = {
        "fills": _surface(
            external_fills,
            ml4t_fills,
            numeric_fields=("quantity", "price", "commission"),
        ),
        "equity": equity_surface,
        "terminal": _surface(terminal_external, terminal_ml4t, numeric_fields=("final_value",)),
    }
    if "num_rejections" in external and "num_rejections" in ml4t:
        rejection_external = [{"count": str(external["num_rejections"])}]
        rejection_ml4t = [{"count": str(ml4t["num_rejections"])}]
        surfaces["rejection_count"] = _surface(
            rejection_external, rejection_ml4t, numeric_fields=("count",)
        )
    if case_study == "etfs" and framework == "lean":
        external_cash = _value_records(external_frames["equity"], field="cash", intraday=intraday)
        ml4t_cash = _value_records(ml4t_frames["portfolio_state"], field="cash", intraday=intraday)
        cash_surface, _, _ = _value_surface(external_cash, ml4t_cash, field="cash")
        surfaces["cash"] = cash_surface

    surface_values = list(surfaces.values())
    passed = all(bool(value["passed"]) for value in surface_values)
    negative_control = [dict(record) for record in ml4t_fills]
    if not negative_control:
        raise ValueError(f"Negative control requires fills for {case_study}/{framework}")
    negative_control[0]["price"] = format(
        Decimal(negative_control[0]["price"]) + CANONICAL_QUANTUM, "f"
    )
    detected = not compare_records(
        external_fills,
        negative_control,
        numeric_fields=("quantity", "price", "commission"),
    )["passed"]
    if not detected:
        raise RuntimeError(f"Negative control was not detected for {case_study}/{framework}")
    return {
        "case_study": case_study,
        "framework": framework,
        "status": "pass" if passed else "fail",
        "input_bundle_sha256": external["input_bundle_sha256"],
        "profile": expected_profile,
        "engine_seconds": {
            "framework": external["engine_seconds"],
            "ml4t": ml4t["engine_seconds"],
            "single_run_diagnostic_only": True,
        },
        "surfaces": surfaces,
        "negative_control": {"mutation": "first fill price + 0.00000001", "detected": detected},
        "excluded_surfaces": {
            "intents": "shared frozen target input is identified, but engines do not expose a common native intent log",
            "accepted_unfilled_orders": "not exposed by every native runner",
            "positions": "not exposed in a common native record schema by every runner",
            "closed_trades": "not a common native framework concept",
            "cash": (
                "compared separately for LEAN equities"
                if case_study == "etfs" and framework == "lean"
                else "not exposed with common accounting semantics"
            ),
        },
        "evidence": {
            "framework_manifest_sha256": _sha256(external_path / "manifest.json"),
            "ml4t_manifest_sha256": _sha256(ml4t_path / "manifest.json"),
        },
    }


def build_report(evidence_root: Path) -> dict[str, Any]:
    """Build the complete required and unsupported real-strategy matrix."""
    applicability_path = VALIDATION_DIR / "real_strategy_applicability.toml"
    applicability = tomllib.loads(applicability_path.read_text(encoding="utf-8"))
    targets = tomllib.loads((VALIDATION_DIR / "framework_targets.toml").read_text())
    adapter_paths = {
        "vectorbt_pro": VALIDATION_DIR / "real_strategy_vectorbt.py",
        "vectorbt_oss": VALIDATION_DIR / "real_strategy_vectorbt.py",
        "backtrader": VALIDATION_DIR / "real_strategy_backtrader.py",
        "zipline": VALIDATION_DIR / "real_strategy_zipline.py",
        "lean_equity": VALIDATION_DIR / "real_strategy_lean.py",
        "lean_crypto": VALIDATION_DIR / "real_strategy_lean_crypto.py",
        "ml4t": VALIDATION_DIR / "real_strategy_runner.py",
        "comparison_input": VALIDATION_DIR / "real_strategy_input.py",
    }
    records: list[dict[str, object]] = []
    for pair in applicability["pair"]:
        if pair["status"] == "required":
            records.append(
                _comparison_record(
                    case_study=pair["case_study"],
                    framework=pair["framework"],
                    evidence_root=evidence_root,
                )
            )
        else:
            records.append(
                {
                    "case_study": pair["case_study"],
                    "framework": pair["framework"],
                    "status": "unsupported",
                    "reason": pair["native_contract"],
                    "source": pair["source"],
                }
            )
    required = [record for record in records if record["status"] != "unsupported"]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": {
            "case_studies": applicability["metadata"]["case_studies"],
            "frameworks": applicability["metadata"]["frameworks"],
            "required_pairs": len(required),
            "unsupported_pairs": len(records) - len(required),
            "real_strategy_equivalence_gate_passed": all(
                record["status"] == "pass" for record in required
            ),
        },
        "comparison_policy": {
            "canonical_quantum": format(CANONICAL_QUANTUM, "f"),
            "rounding": "ROUND_HALF_EVEN",
            "meaning": "zero numeric gap after canonical quantization, not bit identity",
            "fill_order": "canonical timestamp, asset, side, quantity, price, commission",
            "timestamp_domain": {
                "etfs": "session date",
                "cme_futures": "session date",
                "crypto_perps_funding": "exact UTC event timestamp",
            },
        },
        "provenance": {
            "ml4t": {
                **_git_identity(PROJECT_ROOT),
                "engine_source_sha256": _tree_digest(PROJECT_ROOT / "src/ml4t/backtest"),
            },
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "applicability_sha256": _sha256(applicability_path),
            "adapters": {framework: _sha256(path) for framework, path in adapter_paths.items()},
            "frameworks": {
                framework: targets["framework"][framework]
                for framework in applicability["metadata"]["frameworks"]
            },
        },
        "records": records,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.evidence_root.resolve())
    write_report(report, args.output.resolve())
    required = [record for record in report["records"] if record["status"] != "unsupported"]
    passed = sum(record["status"] == "pass" for record in required)
    print(f"Real-strategy parity: {passed}/{len(required)} required pairs pass")
    return 0 if passed == len(required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
