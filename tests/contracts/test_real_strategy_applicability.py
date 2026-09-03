"""Contracts for the bounded real-strategy framework matrix."""

from __future__ import annotations

import tomllib
from itertools import product
from pathlib import Path


def test_every_selected_case_framework_pair_has_one_disposition() -> None:
    path = Path(__file__).parents[2] / "validation" / "real_strategy_applicability.toml"
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    pairs = payload["pair"]

    expected = set(product(metadata["case_studies"], metadata["frameworks"]))
    actual = {(pair["case_study"], pair["framework"]) for pair in pairs}

    assert len(pairs) == len(actual)
    assert actual == expected
    assert {pair["status"] for pair in pairs} == {"required", "unsupported"}
    assert sum(pair["status"] == "required" for pair in pairs) == 17
    assert sum(pair["status"] == "unsupported" for pair in pairs) == 8
    assert all(pair["native_contract"] for pair in pairs)
    assert all(pair["source"].startswith("https://") for pair in pairs)
