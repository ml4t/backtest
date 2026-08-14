"""Contracts for evidence-derived parity documentation."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_VALIDATION_DIR = Path(__file__).parents[2] / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

import generate_parity_claims  # noqa: E402


def test_legacy_accepted_evidence_cannot_generate_current_claims() -> None:
    before = {path: path.read_bytes() for path in generate_parity_claims.TARGETS}

    with pytest.raises(ValueError, match="Unsupported correctness schema: 1"):
        generate_parity_claims.synchronize(check=False)

    assert {path: path.read_bytes() for path in generate_parity_claims.TARGETS} == before


def test_every_claim_target_uses_the_same_generated_block() -> None:
    blocks = []
    for path in generate_parity_claims.TARGETS:
        document = path.read_text(encoding="utf-8")
        blocks.append(
            document.split(generate_parity_claims.START_MARKER, 1)[1].split(
                generate_parity_claims.END_MARKER, 1
            )[0]
        )

    assert len(set(blocks)) == 1


def test_claims_pin_every_advertised_framework_and_expose_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correctness = generate_parity_claims._load_json(generate_parity_claims.CORRECTNESS_EVIDENCE)
    correctness = copy.deepcopy(correctness)
    for framework in generate_parity_claims.SCENARIO_FRAMEWORKS:
        framework_records = [
            record for record in correctness["records"] if record["framework"] == framework
        ]
        if len(framework_records) < len(generate_parity_claims.SCENARIOS):
            added = copy.deepcopy(framework_records[-1])
            added["scenario_id"] = "17"
            added["scenario_name"] = "High Event Count"
            added["required"] = True
            added["status"] = "pass"
            correctness["records"].append(added)
    large_scale = generate_parity_claims._load_json(generate_parity_claims.LARGE_SCALE_EVIDENCE)
    monkeypatch.setattr(generate_parity_claims, "correctness_report_failures", lambda _: [])

    claims = generate_parity_claims.render_claims(correctness, large_scale)

    for pin in correctness["frameworks"].values():
        assert pin["version"] in claims
        assert pin["source"] in claims
        assert f"`{pin['profile']}`" in claims
    assert "16/16 exact" in claims
    assert "225,844 trades" in claims
    assert "No large-scale claim is published for Backtrader" in claims

    failed_correctness = copy.deepcopy(correctness)
    failed_record = next(
        record
        for record in failed_correctness["records"]
        if record["framework"] == "zipline" and record["scenario_id"] == "15"
    )
    failed_record["status"] = "comparison_failure"
    failed_claims = generate_parity_claims.render_claims(failed_correctness, large_scale)
    assert "15/16 pass; blocked by scenario 15" in failed_claims
