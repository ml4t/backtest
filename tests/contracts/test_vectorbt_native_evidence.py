"""Contracts for retained framework-native VectorBT evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_ORACLE = _ROOT / "validation/native/vectorbt_behavior.py"
_EVIDENCE = _ROOT / "validation/native/evidence"

EXPECTED_CHECKS = {
    "accumulation",
    "cash_sharing_and_call_sequence",
    "cash_sharing_short_collateral",
    "defaults",
    "explicit_fill_price",
    "fees_and_slippage",
    "insufficient_cash_partial_fill",
    "integer_signal_dtype",
    "long_signal_conflict",
    "missing_order_price",
    "record_construction",
    "short_cash",
    "signal_timing_and_default_fill",
    "stop_fill",
    "target_percent_sizing",
    "trailing_stop_extreme_and_intrabar_fill",
}


def test_retained_vectorbt_native_evidence_is_complete_and_current() -> None:
    oracle_sha256 = hashlib.sha256(_ORACLE.read_bytes()).hexdigest()
    identities = {
        "vectorbt_oss-1.1.0.json": (
            "vectorbt_oss",
            "1.1.0",
            "259d2d89fe2e7638baf3ca76c394937cd32b656d",
        ),
        "vectorbt_pro-2026.6.27.json": (
            "vectorbt_pro",
            "2026.6.27",
            "6e18cf0aa37849cfc20848f40f1d26ecfdc771b4",
        ),
    }

    for filename, (framework, version, commit) in identities.items():
        payload = json.loads((_EVIDENCE / filename).read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["framework"] == framework
        assert payload["version"] == version
        assert payload["source_commit"] == commit
        assert payload["oracle_sha256"] == oracle_sha256
        assert payload["passed"] is True
        assert {check["id"] for check in payload["checks"]} == EXPECTED_CHECKS
        assert all(check["passed"] for check in payload["checks"])
        assert all(check["actual"] == check["expected"] for check in payload["checks"])
        for location in payload["source_locations"].values():
            assert commit in location["source_url"]
            assert location["path"].endswith("portfolio/base.py")
