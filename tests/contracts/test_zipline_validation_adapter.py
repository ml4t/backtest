"""Contracts for the Zipline release-validation adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_VALIDATION_DIR = Path(__file__).parents[2] / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

from frameworks.zipline import _activate_reconciled_position  # noqa: E402


def test_risk_basis_activates_from_reconciled_fill() -> None:
    context = SimpleNamespace(entry_price=None, high_water_mark=None)
    pending = SimpleNamespace(amount=0, cost_basis=0.0)
    filled = SimpleNamespace(amount=100, cost_basis=96.7)

    _activate_reconciled_position(context, pending)
    assert context.entry_price is None
    assert context.high_water_mark is None

    _activate_reconciled_position(context, filled)
    assert context.entry_price == 96.7
    assert context.high_water_mark == 96.7
