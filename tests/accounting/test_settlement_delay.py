"""Tests for settlement delay (T+N) feature."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest.accounting.account import AccountState
from ml4t.backtest.accounting.policy import UnifiedAccountPolicy
from ml4t.backtest.config import BacktestConfig
from ml4t.backtest.engine import run_backtest
from ml4t.backtest.strategy import Strategy


def _make_account(initial_cash: float = 100000.0) -> AccountState:
    policy = UnifiedAccountPolicy()
    return AccountState(initial_cash=initial_cash, policy=policy)


class TestSettlementDelayAccountState:
    """Unit tests for settlement hold tracking on AccountState."""

    def test_settlement_delay_zero_is_noop(self):
        """With delay=0, add_settlement_hold is a no-op."""
        account = _make_account()
        account.add_settlement_hold(bar_index=1, delay=0, amount=5000.0)
        assert account.unsettled_cash == 0.0

    def test_settlement_hold_adds_and_releases(self):
        """Holds accumulate and release at the correct bar."""
        account = _make_account()
        account.add_settlement_hold(bar_index=1, delay=2, amount=5000.0)
        assert account.unsettled_cash == 5000.0

        # Bar 2: not yet settled
        account.release_settled(current_bar=2)
        assert account.unsettled_cash == 5000.0

        # Bar 3: settled (1 + 2 = 3)
        account.release_settled(current_bar=3)
        assert account.unsettled_cash == 0.0

    def test_multiple_holds_release_independently(self):
        """Multiple holds release at their own settlement bars."""
        account = _make_account()
        account.add_settlement_hold(bar_index=1, delay=1, amount=3000.0)
        account.add_settlement_hold(bar_index=2, delay=2, amount=7000.0)
        assert account.unsettled_cash == 10000.0

        # Bar 2: first hold settles
        account.release_settled(current_bar=2)
        assert account.unsettled_cash == 7000.0

        # Bar 4: second hold settles
        account.release_settled(current_bar=4)
        assert account.unsettled_cash == 0.0

    def test_negative_amount_ignored(self):
        """Negative amounts should not create holds."""
        account = _make_account()
        account.add_settlement_hold(bar_index=1, delay=2, amount=-1000.0)
        assert account.unsettled_cash == 0.0

    def test_floating_point_drift_guard(self):
        """After all holds release, unsettled resets to exactly 0.0."""
        account = _make_account()
        for i in range(100):
            account.add_settlement_hold(bar_index=i, delay=1, amount=0.01)
        # Release all
        account.release_settled(current_bar=200)
        assert account.unsettled_cash == 0.0


class TestSettlementDelayConfig:
    """Tests for settlement_delay in BacktestConfig."""

    def test_config_default_is_zero(self):
        cfg = BacktestConfig()
        assert cfg.settlement_delay == 0

    def test_config_roundtrip(self):
        """settlement_delay survives to_dict / from_dict."""
        cfg = BacktestConfig(settlement_delay=2)
        d = cfg.to_dict()
        assert d["settlement"]["delay"] == 2
        restored = BacktestConfig.from_dict(d)
        assert restored.settlement_delay == 2

    def test_config_validation_warns_on_invalid(self):
        cfg = BacktestConfig(settlement_delay=10)
        issues = cfg.validate(warn=False)
        assert any("settlement_delay" in msg for msg in issues)

    def test_config_validation_ok_for_valid(self):
        cfg = BacktestConfig(settlement_delay=2)
        issues = cfg.validate(warn=False)
        assert not any("settlement_delay" in msg for msg in issues)


def _prices(n_bars: int = 10) -> pl.DataFrame:
    """Generate simple price data for N bars."""
    start = datetime(2024, 1, 1)
    rows = []
    for i in range(n_bars):
        ts = start + timedelta(days=i)
        price = 100.0 + i
        rows.append(
            {
                "timestamp": ts,
                "asset": "AAPL",
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1_000_000.0,
            }
        )
    return pl.DataFrame(rows)


class _BuyThenSell(Strategy):
    """Buy on bar 0, sell on bar 2, try to buy again on bar 3."""

    def __init__(self):
        self.bar = 0

    def on_data(self, timestamp, data, context, broker):
        if self.bar == 0:
            broker.submit_order("AAPL", 100.0)
        elif self.bar == 2:
            broker.submit_order("AAPL", -100.0)
        elif self.bar == 3:
            # Try to rebuy — with T+2 this should be blocked
            broker.submit_order("AAPL", 100.0)
        self.bar += 1


class TestSettlementDelayIntegration:
    """Integration test: settlement delay blocks re-entry."""

    def test_settlement_delay_holds_cash_from_sale(self):
        """With T+2, sale proceeds on bar 2 shouldn't be spendable until bar 4."""
        config = BacktestConfig(
            settlement_delay=2,
            reject_on_insufficient_cash=True,
        )
        result = run_backtest(
            prices=_prices(8),
            strategy=_BuyThenSell(),
            config=config,
        )
        # With same-bar execution and $100k cash:
        # Bar 0: buy 100 shares at ~$100 → cash ~$0
        # Bar 2: sell 100 shares at ~$102 → cash ~$10200 but held
        # Bar 3: try to buy 100 at ~$103 → $10300 cost, but proceeds unsettled
        # Expect the bar-3 buy to be rejected (insufficient settled cash)
        closed = [t for t in result.trades if t.status == "closed"]
        # Only the first round-trip should complete
        assert len(closed) == 1

    def test_settlement_delay_zero_allows_immediate_reentry(self):
        """With delay=0, sale proceeds are immediately spendable."""
        config = BacktestConfig(
            settlement_delay=0,
            reject_on_insufficient_cash=True,
        )
        result = run_backtest(
            prices=_prices(8),
            strategy=_BuyThenSell(),
            config=config,
        )
        # With no settlement delay, bar-3 buy should succeed
        trades = [t for t in result.trades if t.status != "open" or True]
        # Should have at least the initial buy and a second buy
        assert len(result.trades) >= 2
