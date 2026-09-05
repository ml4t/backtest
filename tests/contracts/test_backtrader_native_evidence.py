"""Contracts for retained framework-native Backtrader evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_ORACLE = _ROOT / "validation/native/backtrader_behavior.py"
_EVIDENCE = _ROOT / "validation/native/evidence/backtrader-1.9.78.123.json"

EXPECTED_CHECKS = {
    "cash_rejection_and_configured_leverage",
    "cheat_on_close",
    "commission_headroom",
    "defaults",
    "final_bar_market_order",
    "gap_reversal_executes_close_leg_despite_margin",
    "integer_target_percent",
    "late_feed_start",
    "losing_short_cover_submission_rejection",
    "missing_bar_uses_last_value",
    "next_bar_open_and_gap",
    "short_cash",
    "signal_price_stop_basis",
    "submission_sequence",
    "trade_record",
    "trailing_stop_signal_close_and_lagged",
}


def test_retained_backtrader_native_evidence_is_complete_and_current() -> None:
    payload = json.loads(_EVIDENCE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["framework"] == "backtrader"
    assert payload["package"] == "backtrader"
    assert payload["version"] == "1.9.78.123"
    assert payload["artifact"] == "backtrader-1.9.78.123-py2.py3-none-any.whl"
    assert payload["artifact_sha256"] == (
        "9a07a516b0de9155539a35c56e9404d8711dd7020b3d37b30495e83e1b9d5dfd"
    )
    assert payload["oracle_sha256"] == hashlib.sha256(_ORACLE.read_bytes()).hexdigest()
    assert payload["passed"] is True
    assert {check["id"] for check in payload["checks"]} == EXPECTED_CHECKS
    assert all(check["passed"] for check in payload["checks"])
    assert all(check["actual"] == check["expected"] for check in payload["checks"])
    for location in payload["source_locations"].values():
        assert location["artifact"] == payload["artifact"]
        assert location["artifact_sha256"] == payload["artifact_sha256"]
        assert location["path"].startswith("backtrader/")
        assert location["source"].endswith("/backtrader/1.9.78.123/")
