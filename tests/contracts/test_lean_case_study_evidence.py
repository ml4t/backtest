"""Contracts for retained fresh LEAN case-study evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_WORKSPACE = _ROOT / "validation/lean/workspace"
_DATA = _WORKSPACE / "data/equity/usa/daily"
_SUPPORT = _ROOT / "validation/lean/support"
_EVIDENCE = _ROOT / "validation/lean/case_study_evidence.json"
_CASES = {
    "chapter16_etfs",
    "chapter16_sp500_equity_option_analytics",
    "chapter16_us_equities_panel",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_compressed(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix == ".xz":
        return lzma.decompress(payload)
    if path.suffix == ".gz":
        return gzip.decompress(payload)
    return payload


def _reconstruct_order_events(paths: list[Path]) -> bytes:
    parts = [_read_compressed(path) for path in paths]
    if len(parts) == 1:
        return parts[0]
    header = parts[0].splitlines(keepends=True)[0]
    body = [parts[0], *(b"".join(part.splitlines(keepends=True)[1:]) for part in parts[1:])]
    assert parts[0].startswith(header)
    return b"".join(body)


def test_retained_lean_case_study_evidence_is_fresh_and_complete() -> None:
    payload = json.loads(_EVIDENCE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["passed"] is True
    assert payload["promoted"] is True
    assert payload["framework"]["version"] == "18001"
    assert payload["framework"]["artifact"].startswith("quantconnect/lean@sha256:")
    assert payload["cli_observed"] == "lean 1.0.228"
    assert {case["case"] for case in payload["cases"]} == _CASES

    for relative, digest in payload["producer_files"].items():
        assert _digest(_ROOT / relative) == digest
    for relative, digest in payload["support_files"].items():
        assert _digest(_SUPPORT / relative) == digest

    for case in payload["cases"]:
        assert case["passed"] is True
        assert case["comparison"]["fill_gap"] == 0
        assert case["comparison"]["sorted_fill_multiset_match"] is True
        assert case["comparison"]["canonical_final_value_match"] is True
        assert case["comparison"]["canonical_final_value_gap_usd"] == 0.0
        assert case["lean_fill_surface_sha256"] == case["ml4t_fill_surface_sha256"]

        project = _WORKSPACE / case["case"]
        for relative, digest in case["inputs"].items():
            path = (
                _DATA / Path(relative).name if relative.startswith("data/") else project / relative
            )
            assert _digest(path) == digest

        retained = [_ROOT / relative for relative in case["retained_artifacts"]]
        for path in retained:
            assert _digest(path) == case["retained_artifacts"][path.relative_to(_ROOT).as_posix()]
        order_paths = [path for path in retained if "order_events" in path.name]
        equity_path = next(path for path in retained if path.name == "ml4t_daily_equity.csv")
        assert (
            hashlib.sha256(_reconstruct_order_events(order_paths)).hexdigest()
            == case["raw_order_events_sha256"]
        )
        assert _digest(equity_path) == case["raw_daily_equity_sha256"]
