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
    correctness = copy.deepcopy(
        generate_parity_claims._load_json(generate_parity_claims.CORRECTNESS_EVIDENCE)
    )
    large_scale = generate_parity_claims._load_json(generate_parity_claims.LARGE_SCALE_EVIDENCE)
    real_strategy = generate_parity_claims._load_json(generate_parity_claims.REAL_STRATEGY_EVIDENCE)
    correctness["schema_version"] = 1

    with pytest.raises(ValueError, match="Unsupported correctness schema: 1"):
        generate_parity_claims._validate_evidence(correctness, large_scale, real_strategy)


def test_generated_claims_are_current() -> None:
    assert generate_parity_claims.synchronize(check=True) == []


def test_claim_generation_rejects_hidden_valuation_dates() -> None:
    correctness = generate_parity_claims._load_json(generate_parity_claims.CORRECTNESS_EVIDENCE)
    large_scale = generate_parity_claims._load_json(generate_parity_claims.LARGE_SCALE_EVIDENCE)
    real_strategy = copy.deepcopy(
        generate_parity_claims._load_json(generate_parity_claims.REAL_STRATEGY_EVIDENCE)
    )
    passing = next(record for record in real_strategy["records"] if record["status"] == "pass")
    passing["surfaces"]["equity"]["coverage"]["framework_only_timestamps"] = 1

    with pytest.raises(ValueError, match="valuation coverage verdict is inconsistent"):
        generate_parity_claims._validate_evidence(correctness, large_scale, real_strategy)


def test_claim_generation_rejects_stale_real_strategy_engine_source() -> None:
    correctness = generate_parity_claims._load_json(generate_parity_claims.CORRECTNESS_EVIDENCE)
    large_scale = generate_parity_claims._load_json(generate_parity_claims.LARGE_SCALE_EVIDENCE)
    real_strategy = copy.deepcopy(
        generate_parity_claims._load_json(generate_parity_claims.REAL_STRATEGY_EVIDENCE)
    )
    real_strategy["provenance"]["ml4t"]["engine_source_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="engine source digest is stale"):
        generate_parity_claims._validate_evidence(correctness, large_scale, real_strategy)


def test_claim_generation_rejects_stale_real_strategy_performance() -> None:
    correctness = generate_parity_claims._load_json(generate_parity_claims.CORRECTNESS_EVIDENCE)
    large_scale = generate_parity_claims._load_json(generate_parity_claims.LARGE_SCALE_EVIDENCE)
    real_strategy = generate_parity_claims._load_json(generate_parity_claims.REAL_STRATEGY_EVIDENCE)
    performance = copy.deepcopy(
        generate_parity_claims._load_json(generate_parity_claims.REAL_STRATEGY_PERFORMANCE)
    )
    performance["correctness_evidence_generated_at"] = "stale"

    with pytest.raises(ValueError, match="references stale correctness evidence"):
        generate_parity_claims._validate_evidence(
            correctness, large_scale, real_strategy, performance
        )


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
    real_strategy = generate_parity_claims._load_json(generate_parity_claims.REAL_STRATEGY_EVIDENCE)
    monkeypatch.setattr(generate_parity_claims, "correctness_report_failures", lambda _: [])
    monkeypatch.setattr(generate_parity_claims, "large_scale_report_failures", lambda _: [])

    claims = generate_parity_claims.render_claims(correctness, large_scale, real_strategy)

    for pin in correctness["frameworks"].values():
        assert pin["version"] in claims
        assert pin["source"] in claims
        assert f"`{pin['profile']}`" in claims
    assert "16/16 exact" in claims
    assert "synthetic stress workload contains 250 assets" in claims
    assert "6/8 supported pairs pass" in claims
    assert "identical valuation timestamp coverage" in claims
    for record in large_scale["frameworks"]:
        target = record["target"]
        checks = generate_parity_claims._comparison_checks(record)
        assert f"`{target['profile']}`" in claims
        assert target["version"] in claims
        assert f"{checks['fills']['expected_count']:,}" in claims
        assert f"{checks['trades']['expected_count']:,}" in claims
    assert "validation/LARGE_SCALE_RESULTS.json" in claims
    assert "No large-scale claim is published" not in claims

    failed_correctness = copy.deepcopy(correctness)
    failed_record = next(
        record
        for record in failed_correctness["records"]
        if record["framework"] == "zipline" and record["scenario_id"] == "15"
    )
    failed_record["status"] = "comparison_failure"
    failed_claims = generate_parity_claims.render_claims(
        failed_correctness, large_scale, real_strategy
    )
    assert "15/16 pass; blocked by scenario 15" in failed_claims
