"""Contracts for retained framework-native LEAN evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_ORACLE = _ROOT / "validation/native/lean_behavior.py"
_PROJECT = _ROOT / "validation/native/lean_project"
_SUPPORT = _ROOT / "validation/lean/support"
_EVIDENCE = _ROOT / "validation/native/evidence/lean-18001.json"

EXPECTED_CHECKS = {
    "buying_power_allowed",
    "buying_power_rejected",
    "buying_power_sequence",
    "default_full_fill",
    "default_models",
    "explicit_costs",
    "fill_forward",
    "final_bar_order",
    "liquidation",
    "submission_sequence",
    "target_sizing",
    "terminal_holding",
    "timing",
}


def test_retained_lean_native_evidence_is_complete_and_current() -> None:
    payload = json.loads(_EVIDENCE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["framework"] == "lean"
    assert payload["engine"]["engine_version"] == "18001"
    assert payload["engine"]["source_commit"] == ("278fcb3d1b815b63ccadba68d7ae54422e34b792")
    assert payload["engine"]["image"] == (
        "quantconnect/lean@sha256:ecd62b0e418d40d1d7c0cd95e90a94e397642a21d2c8810614830c8a4e9a8f70"
    )
    assert payload["engine"]["platform_artifact"] == (
        "linux/amd64@sha256:cbe3f26b3f16c57be836b2cf913253d434f58e010c84ed038360d26b9df88307"
    )
    assert payload["cli"] == {
        "immutable_id": ("sha256:eaa4c08f16295b76f005e429d9ca0d0453784dc1a40c1f5cbe5e50c02a05bd7c"),
        "package": "lean",
        "version": "1.0.228",
    }
    assert payload["models"] == {
        "account_type": "Margin",
        "brokerage": "DefaultBrokerageModel",
        "security": "Equity USA, daily, adjusted normalization, leverage 2",
    }
    assert payload["oracle_sha256"] == hashlib.sha256(_ORACLE.read_bytes()).hexdigest()
    expected_files = {
        path.relative_to(_PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_PROJECT.rglob("*"))
        if path.is_file()
    }
    assert payload["oracle_files"] == expected_files
    expected_support = {
        path.relative_to(_SUPPORT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_SUPPORT.rglob("*"))
        if path.is_file()
    }
    assert payload["support_files"] == expected_support
    assert payload["passed"] is True
    assert {check["id"] for check in payload["checks"]} == EXPECTED_CHECKS
    assert all(check["passed"] for check in payload["checks"])
    assert all(check["contract_actual"] == check["expected"] for check in payload["checks"])
    assert all(
        payload["engine"]["source_commit"] in source
        for source in payload["source_references"].values()
    )
