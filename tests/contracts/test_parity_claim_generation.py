"""Contracts for evidence-derived parity documentation."""

from __future__ import annotations

import sys
from pathlib import Path

_VALIDATION_DIR = Path(__file__).parents[2] / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

import generate_parity_claims  # noqa: E402


def test_parity_claim_documents_match_retained_evidence() -> None:
    assert generate_parity_claims.synchronize(check=True) == []


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


def test_claims_pin_every_advertised_framework_and_expose_failures() -> None:
    correctness = generate_parity_claims._load_json(generate_parity_claims.CORRECTNESS_EVIDENCE)
    large_scale = generate_parity_claims._load_json(generate_parity_claims.LARGE_SCALE_EVIDENCE)

    claims = generate_parity_claims.render_claims(correctness, large_scale)

    for pin in correctness["frameworks"].values():
        assert pin["version"] in claims
        assert pin["source"] in claims
        assert f"`{pin['profile']}`" in claims
    assert "14/15 pass; blocked by scenario 15" in claims
    assert "225,844 trades" in claims
    assert "No large-scale claim is published for Backtrader" in claims
