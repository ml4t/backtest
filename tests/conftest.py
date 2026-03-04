"""Root conftest.py — always-on accounting invariants for ml4t-backtest.

Wraps Engine.run() so that every test exercising the engine automatically
gets universal invariant checks. Zero per-test effort; catches accounting
regressions the moment they happen.
"""

from __future__ import annotations

import pytest

from ml4t.backtest import Engine

from .helpers.invariants import assert_result_invariants


@pytest.fixture(autouse=True)
def _patch_engine_invariants(request, monkeypatch):
    """Monkeypatch Engine.run() to assert result invariants after every call.

    Opt-out by marking a test with @pytest.mark.no_invariant_check.
    """
    if "no_invariant_check" in request.keywords:
        return

    original_run = Engine.run

    def checked_run(self):
        result = original_run(self)
        initial_cash = self.config.initial_cash if self.config else 100_000.0
        assert_result_invariants(result, initial_cash)
        return result

    monkeypatch.setattr(Engine, "run", checked_run)
