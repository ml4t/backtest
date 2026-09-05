"""Contracts for retained native and configured Zipline evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_VALIDATION = _ROOT / "validation"
if str(_VALIDATION) not in sys.path:
    sys.path.insert(0, str(_VALIDATION))

from common.provenance import comparison_protocol_metadata  # noqa: E402
from scenarios.definitions import SCENARIOS  # noqa: E402

_ORACLE = _ROOT / "validation/native/zipline_behavior.py"
_EVIDENCE = _ROOT / "validation/native/evidence/zipline-3.1.1.json"

EXPECTED_CHECKS = {
    "cash_and_short_proceeds",
    "configured_next_bar_open_fill",
    "default_next_bar_close_fill",
    "defaults",
    "explicit_minimum_commission",
    "final_bar_market_order",
    "integer_target_percent",
    "missing_bar",
    "session_calendar",
    "submission_sequence",
    "target_percent_snapshot",
    "transaction_records_not_native_trades",
    "volume_share_partial_fills",
}


def test_retained_zipline_native_evidence_is_complete_and_current() -> None:
    payload = json.loads(_EVIDENCE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["framework"] == "zipline"
    assert payload["package"] == "zipline-reloaded"
    assert payload["version"] == "3.1.1"
    assert payload["source_commit"] == "09885a2ebc7567d40942c891b3879dc03c745070"
    assert payload["artifact"] == "zipline_reloaded-3.1.1.tar.gz"
    assert payload["artifact_sha256"] == (
        "4a305524616f7aad836f929e5a2ba5afc7db0e238757f47eb49487d9e2457a6f"
    )
    assert payload["oracle_sha256"] == hashlib.sha256(_ORACLE.read_bytes()).hexdigest()
    assert payload["passed"] is True
    assert {check["id"] for check in payload["checks"]} == EXPECTED_CHECKS
    assert all(check["passed"] for check in payload["checks"])
    assert all(check["actual"] == check["expected"] for check in payload["checks"])
    for location in payload["source_locations"].values():
        assert location["artifact"] == payload["artifact"]
        assert location["artifact_sha256"] == payload["artifact_sha256"]
        assert payload["source_commit"] in location["source_url"]


def test_zipline_risk_scenarios_are_labeled_as_adapter_emulation() -> None:
    for scenario in SCENARIOS.values():
        protocol = comparison_protocol_metadata(scenario, "zipline")
        expected = "adapter_emulated_daily_ohlc" if scenario.risk_rules else "none"
        assert protocol["risk_rules"] == expected
        assert protocol["trade_records"] == "adapter_reconstructed_from_native_transactions"
