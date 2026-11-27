from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, SlippageType


@pytest.mark.parametrize("direction", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("gap", [110.0, 90.0], ids=["up", "down"])
@pytest.mark.parametrize("exit_size", [1, 2, 3], ids=["partial", "full", "reverse"])
def test_realized_excursions_include_the_exit_fill(direction, gap, exit_size):
    times = [datetime(2026, 1, 5) + timedelta(days=i) for i in range(4)]

    class RoundTrip(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("ETF", 2 * direction)
            elif timestamp == times[2]:
                broker.submit_order("ETF", -exit_size * direction)

    prices = [100.0] * 3 + [gap]
    bars = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["ETF"] * 4,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
        }
    )
    result = Engine(
        DataFeed(prices_df=bars),
        RoundTrip(),
        BacktestConfig(
            calendar=None,
            allow_short_selling=True,
            slippage_type=SlippageType.PERCENTAGE,
            slippage_rate=0.0005,
            commission_type=CommissionType.PERCENTAGE,
            commission_rate=0.0005,
        ),
    ).run()
    (trade,) = (t for t in result.trades if t.status in {"closed", "partial"})
    assert trade.exit_time == times[-1] and trade.fees > 0
    expected = direction * (trade.exit_price - trade.entry_price) / trade.entry_price
    assert abs(trade.pnl_percent - expected) < 1e-12
    assert trade.status == ("partial" if exit_size == 1 else "closed")
    assert trade.mae <= expected <= trade.mfe
