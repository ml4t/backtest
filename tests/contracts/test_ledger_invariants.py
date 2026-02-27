from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest.engine import run_backtest
from ml4t.backtest.strategy import Strategy


def _prices(closes: list[float]) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows = []
    for i, close in enumerate(closes):
        ts = start + timedelta(days=i)
        rows.append(
            {
                "timestamp": ts,
                "asset": "AAPL",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    return pl.DataFrame(rows)


class _NoopStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker) -> None:
        return


class _SingleRoundTrip(Strategy):
    def __init__(self) -> None:
        self.bar = 0

    def on_data(self, timestamp, data, context, broker) -> None:
        self.bar += 1
        if self.bar == 1:
            broker.submit_order("AAPL", 10.0)
        elif self.bar == 3:
            broker.close_position("AAPL")


def test_no_trade_preserves_cash_and_equity() -> None:
    result = run_backtest(prices=_prices([100.0, 101.0, 102.0]), strategy=_NoopStrategy())

    assert result.metrics["initial_cash"] == 100000.0
    assert result.metrics["final_value"] == 100000.0
    assert result.metrics["num_trades"] == 0
    assert len(result.trades) == 0
    assert len(result.fills) == 0
    assert all(eq == 100000.0 for _, eq in result.equity_curve)


def test_closed_trade_pnl_reconciles_to_final_value() -> None:
    result = run_backtest(prices=_prices([100.0, 110.0, 120.0, 130.0]), strategy=_SingleRoundTrip())

    closed_trades = [t for t in result.trades if t.status == "closed"]
    assert len(closed_trades) == 1

    total_pnl = sum(t.pnl for t in closed_trades)
    initial_cash = result.metrics["initial_cash"]
    final_value = result.metrics["final_value"]

    assert abs((initial_cash + total_pnl) - final_value) < 1e-9
    assert abs(result.equity_curve[-1][1] - final_value) < 1e-9
