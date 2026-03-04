"""Self-consistency tests for the reference oracle.

These verify the oracle itself is correct before we use it as a reference.
"""

from __future__ import annotations

import pytest

from .engine import FillTiming, OracleBar, OracleFillRule, OracleSignal, run_oracle


class TestOracleLongRoundTrips:
    """Verify oracle computes correct values for long trades."""

    def test_profitable_long_no_costs(self):
        bars = [OracleBar(100, 105, 95, 100), OracleBar(110, 115, 105, 110)]
        signals = [
            OracleSignal(0, "long", "entry", 100),
            OracleSignal(1, "long", "exit", 100),
        ]
        result = run_oracle(bars, signals)

        assert len(result.trades) == 1
        t = result.trades[0]
        assert t.gross_pnl == pytest.approx(1000.0)  # (110-100)*100
        assert t.fees == 0.0
        assert t.net_pnl == pytest.approx(1000.0)
        assert t.pnl_percent == pytest.approx(0.10)   # 10%
        assert result.final_cash == pytest.approx(101_000.0)

    def test_losing_long_no_costs(self):
        bars = [OracleBar(100, 105, 95, 100), OracleBar(90, 95, 85, 90)]
        signals = [
            OracleSignal(0, "long", "entry", 100),
            OracleSignal(1, "long", "exit", 100),
        ]
        result = run_oracle(bars, signals)

        t = result.trades[0]
        assert t.gross_pnl == pytest.approx(-1000.0)
        assert t.pnl_percent == pytest.approx(-0.10)
        assert result.final_cash == pytest.approx(99_000.0)

    def test_long_with_commission(self):
        bars = [OracleBar(100, 100, 100, 100), OracleBar(110, 110, 110, 110)]
        signals = [
            OracleSignal(0, "long", "entry", 100),
            OracleSignal(1, "long", "exit", 100),
        ]
        rule = OracleFillRule(commission_rate=0.001)
        result = run_oracle(bars, signals, rule)

        t = result.trades[0]
        entry_comm = 0.001 * 100 * 100  # 10
        exit_comm = 0.001 * 110 * 100   # 11
        assert t.fees == pytest.approx(entry_comm + exit_comm)
        assert t.gross_pnl == pytest.approx(1000.0)
        assert t.net_pnl == pytest.approx(1000.0 - 21.0)

    def test_breakeven_long(self):
        bars = [OracleBar(100, 100, 100, 100), OracleBar(100, 100, 100, 100)]
        signals = [
            OracleSignal(0, "long", "entry", 50),
            OracleSignal(1, "long", "exit", 50),
        ]
        result = run_oracle(bars, signals)

        t = result.trades[0]
        assert t.gross_pnl == pytest.approx(0.0)
        assert t.pnl_percent == pytest.approx(0.0)


class TestOracleShortRoundTrips:
    """Verify oracle computes correct values for short trades."""

    def test_profitable_short_no_costs(self):
        bars = [OracleBar(100, 105, 95, 100), OracleBar(90, 95, 85, 90)]
        signals = [
            OracleSignal(0, "short", "entry", 100),
            OracleSignal(1, "short", "exit", 100),
        ]
        result = run_oracle(bars, signals)

        t = result.trades[0]
        assert t.direction == "short"
        assert t.gross_pnl == pytest.approx(1000.0)  # (100-90)*100
        assert t.pnl_percent == pytest.approx(0.10)
        assert result.final_cash == pytest.approx(101_000.0)

    def test_losing_short_no_costs(self):
        bars = [OracleBar(100, 105, 95, 100), OracleBar(110, 115, 105, 110)]
        signals = [
            OracleSignal(0, "short", "entry", 100),
            OracleSignal(1, "short", "exit", 100),
        ]
        result = run_oracle(bars, signals)

        t = result.trades[0]
        assert t.gross_pnl == pytest.approx(-1000.0)
        assert t.pnl_percent == pytest.approx(-0.10)
        assert result.final_cash == pytest.approx(99_000.0)

    def test_short_with_commission(self):
        bars = [OracleBar(100, 100, 100, 100), OracleBar(90, 90, 90, 90)]
        signals = [
            OracleSignal(0, "short", "entry", 100),
            OracleSignal(1, "short", "exit", 100),
        ]
        rule = OracleFillRule(commission_rate=0.001)
        result = run_oracle(bars, signals, rule)

        t = result.trades[0]
        entry_comm = 0.001 * 100 * 100  # 10
        exit_comm = 0.001 * 90 * 100    # 9
        assert t.fees == pytest.approx(entry_comm + exit_comm)
        assert t.gross_pnl == pytest.approx(1000.0)
        assert t.net_pnl == pytest.approx(1000.0 - 19.0)


class TestOracleCashConservation:
    """Verify cash is conserved across all trades."""

    def test_cash_conservation_long(self):
        bars = [
            OracleBar(100, 100, 100, 100),
            OracleBar(110, 110, 110, 110),
        ]
        signals = [
            OracleSignal(0, "long", "entry", 100),
            OracleSignal(1, "long", "exit", 100),
        ]
        rule = OracleFillRule(commission_rate=0.001)
        result = run_oracle(bars, signals, rule, initial_cash=50_000)

        expected = 50_000 + result.trades[0].net_pnl
        assert result.final_cash == pytest.approx(expected, abs=1e-6)

    def test_cash_conservation_short(self):
        bars = [
            OracleBar(100, 100, 100, 100),
            OracleBar(90, 90, 90, 90),
        ]
        signals = [
            OracleSignal(0, "short", "entry", 100),
            OracleSignal(1, "short", "exit", 100),
        ]
        rule = OracleFillRule(commission_rate=0.002)
        result = run_oracle(bars, signals, rule, initial_cash=50_000)

        expected = 50_000 + result.trades[0].net_pnl
        assert result.final_cash == pytest.approx(expected, abs=1e-6)

    def test_no_signals_preserves_cash(self):
        bars = [OracleBar(100, 100, 100, 100)]
        result = run_oracle(bars, [], initial_cash=12345.0)
        assert result.final_cash == 12345.0
        assert len(result.trades) == 0


class TestOracleEdgeCases:
    """Edge cases and boundary conditions."""

    def test_unmatched_exit_is_ignored(self):
        bars = [OracleBar(100, 100, 100, 100)]
        signals = [OracleSignal(0, "long", "exit", 100)]
        result = run_oracle(bars, signals)
        assert len(result.trades) == 0
        assert result.final_cash == 100_000.0

    def test_duplicate_entry_is_ignored(self):
        bars = [
            OracleBar(100, 100, 100, 100),
            OracleBar(105, 105, 105, 105),
            OracleBar(110, 110, 110, 110),
        ]
        signals = [
            OracleSignal(0, "long", "entry", 100),
            OracleSignal(1, "long", "entry", 100),  # Ignored (already in position)
            OracleSignal(2, "long", "exit", 100),
        ]
        result = run_oracle(bars, signals)
        assert len(result.trades) == 1

    def test_next_bar_timing(self):
        bars = [
            OracleBar(100, 100, 100, 100),
            OracleBar(102, 105, 100, 103),
            OracleBar(108, 110, 107, 109),
        ]
        signals = [
            OracleSignal(0, "long", "entry", 100),
            OracleSignal(1, "long", "exit", 100),
        ]
        rule = OracleFillRule(timing=FillTiming.NEXT_BAR)
        result = run_oracle(bars, signals, rule)

        t = result.trades[0]
        # Entry: bar 0 signal → fills at bar 1 open = 102
        assert t.entry_price == pytest.approx(102.0)
        # Exit: bar 1 signal → fills at bar 2 open = 108
        assert t.exit_price == pytest.approx(108.0)
        assert t.gross_pnl == pytest.approx((108 - 102) * 100)
